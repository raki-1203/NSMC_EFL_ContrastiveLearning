import os
import pandas as pd
import numpy as np
import torch

from argparse import ArgumentParser
from glob import glob
from datasets import Features, Value, Dataset
from konlpy.tag import Mecab
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from kss import split_sentences
from transformers import BertTokenizer, DataCollatorWithPadding

from utils.label_descriptions import efl_sentiment_label_descriptions, efl_category_label_descriptions
from utils.model import EFLContrastiveLearningModel


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--task', type=str, default='category', choices=('sentiment', 'category'))

    parser.add_argument('--method', type=str, default='efl_scl', choices=('efl', 'efl_scl', 'std', 'std_scl'))
    parser.add_argument('--model_name_or_path', type=str, default='./model/checkpoint-2000000')
    parser.add_argument('--category_saved_model_path', type=str,
                        default='./model/saved_model/all_category_model_7_3/STEP_700_efl_scl_TASKcategory_LR5e-05_WD0.1_LAMBDA0.1_POOLERcls_TEMP0.1_ACC0.7056')
    parser.add_argument('--sentiment_saved_model_path', type=str,
                        default='./model/saved_model/sentiment_model_7_3/STEP_400_efl_scl_TASKsentiment_LR5e-05_WD0.1_LAMBDA0.9_POOLERcls_TEMP0.05_ACC0.8278')
    parser.add_argument('--vocab_path', type=str, default='./tokenizer/version_1.9')
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pooler_option', type=str, default='cls')

    parser.add_argument('--path_to_test_data', type=str, default='./data/cs_sharing/test')

    args = parser.parse_args()

    if args.device == '0':
        args.device = torch.device('cuda:0')
    if args.device == '1':
        args.device = torch.device('cuda:1')

    print(args)

    return args


def get_efl_predict_dataloader(args, task, tokenizer, mecab, sents):
    if task == 'sentiment':
        efl_predict_data = []
        for sent in sents:
            for category, label_description in efl_sentiment_label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                efl_predict_data.append(new_example)

        efl_predict_df = pd.DataFrame(efl_predict_data)
        # batch_size = len(efl_sentiment_label_descriptions)
    else:
        efl_predict_data = []
        for sent in sents:
            for category, label_description in efl_category_label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                efl_predict_data.append(new_example)

        efl_predict_df = pd.DataFrame(efl_predict_data)
        # batch_size = len(efl_category_label_descriptions)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'sent2': Value(dtype='string', id=None)})

    predict_dataset = Dataset.from_pandas(efl_predict_df, features=f)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def preprocess(example):
        s1 = mecab.morphs(example['sent1'])
        s1 = " ".join(s1)
        s2 = mecab.morphs(example['sent2'])
        s2 = " ".join(s2)
        texts = (s1, s2)
        result = tokenizer(*texts,
                           return_token_type_ids=False,
                           padding=True,
                           truncation=True
                           )

        return result

    predict_dataset = predict_dataset.map(
        preprocess,
        batched=False,
        remove_columns=predict_dataset.column_names,
    )

    return DataLoader(predict_dataset,
                      collate_fn=data_collator,
                      shuffle=False,
                      batch_size=args.batch_size)


def predict(args, tokenizer, sentiment_model, category_model, mecab, text):
    sentiment_model.eval()
    category_model.eval()

    sents = split_sentences(text)

    num_sentiment_label_descriptions = len(efl_sentiment_label_descriptions)
    num_category_label_descriptions = len(efl_category_label_descriptions)

    sentiment_dataloader = get_efl_predict_dataloader(args, 'sentiment', tokenizer, mecab, sents)

    # 불만/일반 감성분석
    sentiment_prediction_probs = []  # [sentence_length * 2, 2]
    with torch.no_grad():
        # 한 문장마다 예측
        for batch in sentiment_dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            output = sentiment_model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'])

            logits = output['logits']

            sentiment_prediction_probs.append(logits.detach().cpu().numpy())

    sentiment_prediction_probs = np.concatenate(sentiment_prediction_probs, axis=0)
    sentiment_prediction_probs = np.reshape(sentiment_prediction_probs, (-1, num_sentiment_label_descriptions, 2))
    sentiment_pos_probs = sentiment_prediction_probs[:, :, 1]
    sentiment_pos_probs = np.reshape(sentiment_pos_probs, (-1, num_sentiment_label_descriptions))
    sentiment_preds = np.argmax(sentiment_pos_probs, axis=-1)

    # 불만인 index 만 추출
    negative_idx = np.where(np.array(sentiment_preds) == 1)[0]
    negative_sents = [sents[idx] for idx in negative_idx]
    num_negative_sents = len(negative_sents)

    category_dataloader = get_efl_predict_dataloader(args, 'category', tokenizer, mecab, negative_sents)

    # 배송/처리/제품/기타 카테고리 분석
    category_prediction_probs = []  # [negative_sentence_length * 3, 2]
    with torch.no_grad():
        # 한 문장마다 예측
        for batch in category_dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            output = category_model(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'])

            logits = output['logits']

            category_prediction_probs.append(logits.detach().cpu().numpy())

    category_prediction_probs = np.concatenate(category_prediction_probs, axis=0)
    category_prediction_probs = np.reshape(category_prediction_probs, (-1, num_category_label_descriptions, 2))

    category_score = [0, 0, 0, 0]
    category_preds = []
    for idx in range(num_negative_sents):
        # flag = False
        # flag_list = [0 if not_entail_prob > entail_prob else 1
        #              for not_entail_prob, entail_prob in category_prediction_probs[idx]]
        # if sum(flag_list) == 0:
        #     flag = True
        #
        # if flag:
        #     pred = 3
        # else:
        #     pred = np.argmax(category_prediction_probs[idx, :, 1].squeeze())

        pred = np.argmax(category_prediction_probs[idx, :, 1].squeeze())

        category_score[pred] += 1
        idx_to_label = {0: 'shipping', 1: 'product', 2: 'processing', 3: 'etc'}
        category_preds.append(idx_to_label[pred])

    category = np.argwhere(category_score == np.amax(category_score))

    return 0, category


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                              do_lower_case=False,
                                              unk_token='<unk>',
                                              sep_token='</s>',
                                              pad_token='<pad>',
                                              cls_token='<s>',
                                              mask_token='<mask>',
                                              model_max_length=args.max_len)

    mecab = Mecab()

    if 'category' == args.task:
        test_data_path = glob(os.path.join(args.path_to_test_data, 'category*.csv'))[0]
        test_df = pd.read_csv(test_data_path)
    elif 'sentiment' == args.task:
        raise NotImplementedError

    sentiment_model = EFLContrastiveLearningModel(args=args)
    sentiment_model_state_dict = torch.load(os.path.join(args.sentiment_saved_model_path, 'model_state_dict.pt'))
    sentiment_model.load_state_dict(sentiment_model_state_dict)
    sentiment_model.to(args.device)

    category_model = EFLContrastiveLearningModel(args=args)
    category_model_state_dict = torch.load(os.path.join(args.category_saved_model_path, 'model_state_dict.pt'))
    category_model.load_state_dict(category_model_state_dict)
    category_model.to(args.device)

    label_to_idx = {'shipping': 0, 'product': 1, 'processing': 2, 'etc': 3}
    pred_list = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        is_positive, category = predict(args, tokenizer, sentiment_model, category_model, mecab, row['text'])
        pred_list.append(category.reshape(category.shape[0]).tolist())

    result_df = pd.DataFrame()
    result_df['sentence'] = test_df['text']
    result_df['label'] = test_df['label'].map(label_to_idx)
    result_df['pred'] = pred_list

    acc = result_df.apply(lambda x: 1 if x['label'] in x['pred'] else 0, axis=1).mean()
    print('Total Accuracy:', acc)

    shipping_acc = result_df[result_df.label == 0].apply(lambda x: 1 if x['label'] in x['pred'] else 0, axis=1).mean()
    print('Shipping Accuracy:', shipping_acc)
    product_acc = result_df[result_df.label == 1].apply(lambda x: 1 if x['label'] in x['pred'] else 0, axis=1).mean()
    print('Product Accuracy:', product_acc)
    processing_acc = result_df[result_df.label == 2].apply(lambda x: 1 if x['label'] in x['pred'] else 0, axis=1).mean()
    print('Processing Accuracy:', processing_acc)
    etc_acc = result_df[result_df.label == 3].apply(lambda x: 1 if x['label'] in x['pred'] else 0, axis=1).mean()
    print('ETC Accuracy:', etc_acc)


if __name__ == '__main__':
    args = get_arguments()

    main(args)

