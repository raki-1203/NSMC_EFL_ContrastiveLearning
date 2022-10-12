import torch
import random
import logging
import transformers
import numpy as np

from argparse import ArgumentParser


class Arguments():

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self):
        self.add_argument('--use_amp', action='store_true')
        self.add_argument('--is_train', action='store_true')
        self.add_argument('--device', type=str,
                          default=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.add_argument('--task', type=str, default='category', choices=('sentiment', 'category'))
        self.add_argument('--trial', type=str, default='0')
        self.add_argument('--write_summary', action='store_true')

    def add_hyper_parameters(self):
        self.add_argument('--method', type=str, default='efl_scl', choices=('efl', 'efl_scl', 'std', 'std_scl'))
        self.add_argument('--model_name_or_path', type=str, default='./model/checkpoint-2000000')
        self.add_argument('--vocab_path', type=str, default='./tokenizer/version_1.9')
        self.add_argument('--max_len', type=int, default=256)
        self.add_argument('--batch_size', type=int, default=64)
        self.add_argument('--epochs', type=int, default=3)
        self.add_argument('--accumulation_steps', type=int, default=1)
        self.add_argument('--eval_steps', type=int, default=250)
        self.add_argument('--seed', type=int, default=12)
        self.add_argument('--lr', type=float, default=0.0005)
        self.add_argument('--weight_decay', type=float, default=0.1)
        self.add_argument('--warmup_ratio', type=float, default=0.05)
        self.add_argument('--temperature', type=float, default=0.05)
        self.add_argument('--pooler_option', type=str, default='cls')
        self.add_argument('--cl_weight', type=float, default=0.9)

    def add_data_parameters(self):
        self.add_argument('--path_to_train_data', type=str, default='nsmc_dataset_ver1')
        self.add_argument('--tensorboard_dir', type=str, default='./tensorboard_logs')
        self.add_argument('--output_path', type=str, default='./model/saved_model')

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:
                print("argparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:
                print("\t", key, ":", value, "\n}")
            else:
                print("\t", key, ":", value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        if args.device == '0':
            args.device = torch.device('cuda:0')
        if args.device == '1':
            args.device = torch.device('cuda:1')

        self.print_args(args)

        return args


class Setting():

    def set_logger(self):
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        _logger.addHandler(stream_handler)
        _logger.setLevel(logging.ERROR)

        transformers.logging.set_verbosity_error()

        return _logger

    def set_seed(self, args):
        seed = args.seed

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run(self):
        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_hyper_parameters()
        parser.add_data_parameters()

        args = parser.parse()
        logger = self.set_logger()
        self.set_seed(args)

        return args, logger
