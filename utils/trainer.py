import os
import sys
import shutil
import torch
import numpy as np

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from konlpy.tag import Mecab

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from utils.data import get_efl_dataloader
from utils.model import EFLContrastiveLearningModel
from utils.loss import RDropSupConLoss
from utils.label_descriptions import efl_sentiment_label_descriptions


class Trainer():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                                       do_lower_case=False,
                                                       unk_token='<unk>',
                                                       sep_token='</s>',
                                                       pad_token='<pad>',
                                                       cls_token='<s>',
                                                       mask_token='<mask>',
                                                       model_max_length=args.max_len)
        self.mecab = Mecab()

        dataloader = get_efl_dataloader(self.args, self.tokenizer, self.mecab)

        self.train_dataloader = dataloader['train']
        self.valid_dataloader = dataloader['valid']

        self.step_per_epoch = len(self.train_dataloader)

        self.model = EFLContrastiveLearningModel(args=args)
        self.model.to(self.args.device)

        self.contrastive_loss = RDropSupConLoss(args.temperature)
        self.supervised_loss = nn.CrossEntropyLoss()

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()

        self.best_valid_acc = 0
        self.best_model_folder = None

        if args.write_summary:
            self.writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir,
                                                             f'{args.method}_TASK{args.task}_LR{args.lr}_WD{args.weight_decay}_LAMBDA{args.cl_weight}_POOLER{args.pooler_option}_TEMP{args.temperature}'))

    def train_epoch(self, epoch):
        self.model.train()

        self.optimizer.zero_grad()

        train_iterator = tqdm(self.train_dataloader, desc='Train Iteration')

        for step, batch in enumerate(train_iterator):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            total_step = epoch * self.step_per_epoch + step

            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                output = self.model(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'])
                p = output['embeddings']  # [CLS] 토큰의 hidden_vector
                logits = output['logits']  # [CLS] 토큰의 last classifier 거친 vector

                if 'scl' in self.args.method:
                    output = self.model(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'])
                    q = output['embeddings']

            preds = torch.argmax(logits, dim=-1)
            ce_loss = self.supervised_loss(logits, batch['ce_label'])

            if 'scl' in self.args.method:
                contrastive_loss = self.contrastive_loss(p, q, batch['scl_label'])
                loss = self.args.cl_weight * contrastive_loss + (1 - self.args.cl_weight) * ce_loss
            else:
                loss = ce_loss

            self.scaler.scale(loss).backward()

            acc = torch.sum(preds.cpu() == batch['ce_label'].cpu())

            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            self.train_loss.update(loss.item(), self.args.batch_size)
            self.train_acc.update(acc.item() / self.args.batch_size)

            if total_step != 0 and total_step % self.args.eval_steps == 0:
                valid_acc, valid_loss = self.validate(self.args)
                self.model.train()
                if self.args.write_summary:
                    self.writer.add_scalar('Loss/train', self.train_loss.avg, total_step)
                    self.writer.add_scalar('Loss/valid', valid_loss, total_step)
                    self.writer.add_scalar('Acc/train', self.train_acc.avg, total_step)
                    self.writer.add_scalar('Acc/valid', valid_acc, total_step)

                print(
                    f'STEP {total_step} | eval loss: {valid_loss:.4f} | eval acc: {valid_acc:.4f} | train loss: {self.train_loss.avg:.4f} | train acc: {self.train_acc.avg:.4f}')
                self.train_loss.reset()
                self.train_acc.reset()

                if valid_acc > self.best_valid_acc:
                    print(f'BEST_BEFORE : {self.best_valid_acc:.4f}, NOW : {valid_acc:.4f}')
                    print(f'Saving Model...')
                    self.best_valid_acc = valid_acc
                    self.save_model(total_step)

    def validate(self, args):

        self.model.eval()

        valid_loss = AverageMeter()

        class_num = len(efl_sentiment_label_descriptions)

        all_prediction_probs = []  # [total_num * class_num, 2]
        all_labels = []  # [total_num * class_num]

        with torch.no_grad():
            valid_iterator = tqdm(self.valid_dataloader, desc='Valid Iteration')
            for step, batch in enumerate(valid_iterator):

                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                output = self.model(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'])

                logits = output['logits']
                preds = torch.argmax(logits, dim=-1)

                efl_label = torch.zeros_like(batch['ce_label'])
                efl_label[batch['ce_label'][0]] = 1
                loss = self.supervised_loss(logits, efl_label)

                valid_loss.update(loss.item(), args.batch_size)

                all_prediction_probs.append(logits.detach().cpu().numpy())
                all_labels.append(batch['ce_label'].detach().cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)
        all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))

        prediction_pos_probs = all_prediction_probs[:, :, 1]
        prediction_pos_probs = np.reshape(prediction_pos_probs, (-1, class_num))
        y_pred_index = np.argmax(prediction_pos_probs, axis=-1)

        y_true_index = np.array([true_label_index for idx, true_label_index in enumerate(all_labels)
                                 if idx % class_num == 0])

        total_num = len(y_true_index)
        correct_num = (y_pred_index == y_true_index).sum()

        valid_loss = valid_loss.avg
        valid_acc = correct_num / total_num

        return valid_acc, valid_loss

    def save_model(self, step):
        if self.best_model_folder:
            shutil.rmtree(self.best_model_folder)

        file_name = f'STEP_{step}_{self.args.method}_TASK{self.args.task}_LR{self.args.lr}_WD{self.args.weight_decay}_LAMBDA{self.args.cl_weight}_POOLER{self.args.pooler_option}_TEMP{self.args.temperature}_ACC{self.best_valid_acc:.4f}'
        output_path = os.path.join(self.args.output_path, file_name)

        os.mkdir(output_path)

        torch.save(self.model.state_dict(), os.path.join(output_path, 'model_state_dict.pt'))

        print(f'Model Saved at {output_path}')
        self.best_model_folder = output_path

    def _get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        return optimizer

    def _get_scheduler(self):
        train_total = self.step_per_epoch * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.args.warmup_ratio * train_total,
                                                    num_training_steps=train_total)
        return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
