import numpy
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
import pandas as pd
from .label_descriptions import efl_category_label_descriptions, scl_label_table, std_label_table, \
    efl_sentiment_label_descriptions, sentiment_scl_label_table, std_sentiment_label_table
import os
from datasets import Features, Value, Dataset
from sklearn.model_selection import train_test_split


def get_std_dataloader(args, tokenizer, mecab):
    if 'sentiment' in args.task:
        dataset = _get_std_cs_sharing_dataset(args)
    else:
        dataset = _get_std_dataset(args)

    train_dataset = dataset['train']
    valid_dataset = dataset['valid']

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def preprocess(example):
        s1 = mecab.morphs(example['sent1'])
        s1 = " ".join(s1)

        texts = (s1,)
        result = tokenizer(*texts,
                           return_token_type_ids=False,
                           padding=True,
                           truncation=True
                           )

        result['ce_label'] = example['ce_label']
        result['scl_label'] = example['scl_label']
        return result

    train_dataset = train_dataset.map(
        preprocess,
        batched=False,
        remove_columns=train_dataset.column_names,
    )
    valid_dataset = valid_dataset.map(
        preprocess,
        batched=False,
        remove_columns=valid_dataset.column_names,
    )

    return {'train': DataLoader(train_dataset,
                                collate_fn=data_collator,
                                shuffle=True,
                                batch_size=args.batch_size),
            'valid': DataLoader(valid_dataset,
                                collate_fn=data_collator,
                                shuffle=False,
                                batch_size=args.batch_size)}


def _get_std_dataset(args):
    train_data = {}
    valid_data = {}

    for category in std_label_table:
        file_name = category + '.txt'
        train_path = os.path.join(args.path_to_train_data, file_name)
        train_data[category] = pd.read_csv(train_path, sep='\t', header=None)
        train_data[category] = train_data[category][0].tolist()

        valid_path = os.path.join(args.path_to_valid_data, file_name)
        valid_data[category] = pd.read_csv(valid_path, sep='\t', header=None)
        valid_data[category] = valid_data[category][0].tolist()

    std_train_data = []
    std_valid_data = []

    for category in std_label_table:
        for sent in train_data[category]:
            new_example = {}
            new_example['sent1'] = sent
            new_example['ce_label'] = std_label_table[category]
            new_example['scl_label'] = std_label_table[category]
            std_train_data.append(new_example)

    for category in std_label_table:
        for sent in valid_data[category]:
            new_example = {}
            new_example['sent1'] = sent
            new_example['ce_label'] = std_label_table[category]
            new_example['scl_label'] = std_label_table[category]
            std_valid_data.append(new_example)

    std_train_data = pd.DataFrame(std_train_data)
    std_valid_data = pd.DataFrame(std_valid_data)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'ce_label': Value(dtype='int8', id=None),
                  'scl_label': Value(dtype='int8', id=None)})

    return {'train': Dataset.from_pandas(std_train_data, features=f),
            'valid': Dataset.from_pandas(std_valid_data, features=f)}


def get_efl_dataloader(args, tokenizer, mecab):
    if 'sentiment' in args.task:
        dataset = _get_efl_cs_sharing_dataset(args)
    else:
        dataset = _get_efl_dataset(args)

    train_dataset = dataset['train']
    valid_dataset = dataset['valid']

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

        result['ce_label'] = example['ce_label']
        result['scl_label'] = example['scl_label']
        return result

    train_dataset = train_dataset.map(
        preprocess,
        batched=False,
        remove_columns=train_dataset.column_names,
    )
    valid_dataset = valid_dataset.map(
        preprocess,
        batched=False,
        remove_columns=valid_dataset.column_names,
    )

    if 'sentiment' in args.task:
        valid_batch_size = len(efl_sentiment_label_descriptions)
    else:
        valid_batch_size = len(efl_category_label_descriptions)

    return {'train': DataLoader(train_dataset,
                                collate_fn=data_collator,
                                shuffle=True,
                                batch_size=args.batch_size),
            'valid': DataLoader(valid_dataset,
                                collate_fn=data_collator,
                                shuffle=False,
                                batch_size=valid_batch_size)}


def _get_efl_dataset(args):
    train_data = {}
    valid_data = {}
    for category in efl_category_label_descriptions:
        file_name = category + '.txt'
        train_path = os.path.join(args.path_to_train_data, file_name)
        train_data[category] = pd.read_csv(train_path, sep='\t', header=None)
        train_data[category] = train_data[category][0].tolist()

        valid_path = os.path.join(args.path_to_valid_data, file_name)
        valid_data[category] = pd.read_csv(valid_path, sep='\t', header=None)
        valid_data[category] = valid_data[category][0].tolist()

    # train_data_length = {key:len(train_data[key]) for key in efl_category_label_descriptions}

    efl_train_data = []
    efl_valid_data = []

    for true_category in efl_category_label_descriptions:
        for sent in train_data[true_category]:
            for category, label_description in efl_category_label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                if category == true_category:
                    new_example['ce_label'] = 1
                else:
                    new_example['ce_label'] = 0

                new_example['scl_label'] = scl_label_table[true_category][category]
                efl_train_data.append(new_example)

        for sent in valid_data[true_category]:
            for category, label_description in efl_category_label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                new_example['ce_label'] = std_label_table[true_category]

                new_example['scl_label'] = scl_label_table[true_category][category]
                efl_valid_data.append(new_example)

    efl_train_data = pd.DataFrame(efl_train_data)
    efl_valid_data = pd.DataFrame(efl_valid_data)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'sent2': Value(dtype='string', id=None),
                  'ce_label': Value(dtype='int8', id=None),
                  'scl_label': Value(dtype='int8', id=None)})

    return {'train': Dataset.from_pandas(efl_train_data, features=f),
            'valid': Dataset.from_pandas(efl_valid_data, features=f)}


def _get_efl_cs_sharing_dataset(args):
    train_data = {}
    valid_data = {}

    train_df = pd.read_csv(args.path_to_train_data)
    valid_df = pd.read_csv(args.path_to_valid_data)

    train_positive_mask = train_df['emotional'].isnull()
    # train_positive_mask = train_df['emotional'] == 'nan'
    train_negative_mask = train_df['emotional'] == '불만'
    valid_positive_mask = valid_df['emotional'].isnull()
    # valid_positive_mask = valid_df['emotional'] == 'nan'
    valid_negative_mask = valid_df['emotional'] == '불만'

    train_data['negative'] = train_df.loc[train_negative_mask]['text'].values.tolist()
    train_data['positive'] = train_df.loc[train_positive_mask]['text'].values.tolist()

    valid_data['negative'] = valid_df.loc[valid_negative_mask]['text'].values.tolist()
    valid_data['positive'] = valid_df.loc[valid_positive_mask]['text'].values.tolist()

    # train_data['negative'], valid_data['negative'] = train_test_split(neg, test_size=0.2, random_state=args.seed)
    # train_data['positive'], valid_data['positive'] = train_test_split(pos, test_size=0.2, random_state=args.seed)

    efl_train_data = []
    efl_valid_data = []

    for true_category in train_data.keys():
        for sent in train_data[true_category]:
            for category, label_description in efl_sentiment_label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                if category == true_category:
                    new_example['ce_label'] = 1
                else:
                    new_example['ce_label'] = 0

                new_example['scl_label'] = sentiment_scl_label_table[true_category][category]
                efl_train_data.append(new_example)

    for true_category in valid_data.keys():
        for sent in valid_data[true_category]:
            for category, label_description in efl_sentiment_label_descriptions.items():
                new_example = {}
                new_example['sent1'] = sent
                new_example['sent2'] = label_description

                new_example['ce_label'] = std_sentiment_label_table[true_category]
                new_example['scl_label'] = sentiment_scl_label_table[true_category][category]
                efl_valid_data.append(new_example)
    efl_train_data = pd.DataFrame(efl_train_data)
    efl_valid_data = pd.DataFrame(efl_valid_data)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'sent2': Value(dtype='string', id=None),
                  'ce_label': Value(dtype='int8', id=None),
                  'scl_label': Value(dtype='int8', id=None)})

    return {'train': Dataset.from_pandas(efl_train_data, features=f),
            'valid': Dataset.from_pandas(efl_valid_data, features=f)}


def _get_std_cs_sharing_dataset(args):
    train_data = {}
    valid_data = {}

    train_df = pd.read_csv(args.path_to_train_data)
    valid_df = pd.read_csv(args.path_to_valid_data)

    train_positive_mask = train_df['emotional'].isnull()
    train_negative_mask = train_df['emotional'] == '불만'
    valid_positive_mask = valid_df['emotional'].isnull()
    valid_negative_mask = valid_df['emotional'] == '불만'

    train_data['negative'] = train_df.loc[train_negative_mask]['text'].values.tolist()
    train_data['positive'] = train_df.loc[train_positive_mask]['text'].values.tolist()

    valid_data['negative'] = valid_df.loc[valid_negative_mask]['text'].values.tolist()
    valid_data['positive'] = valid_df.loc[valid_positive_mask]['text'].values.tolist()

    std_train_data = []
    std_valid_data = []

    for true_category in train_data.keys():
        for sent in train_data[true_category]:
            new_example = {}
            new_example['sent1'] = sent
            new_example['ce_label'] = std_sentiment_label_table[true_category]
            new_example['scl_label'] = std_sentiment_label_table[true_category]
            std_train_data.append(new_example)

    for true_category in valid_data.keys():
        for sent in valid_data[true_category]:
            new_example = {}
            new_example['sent1'] = sent
            new_example['ce_label'] = std_sentiment_label_table[true_category]
            new_example['scl_label'] = std_sentiment_label_table[true_category]
            std_valid_data.append(new_example)

    std_train_data = pd.DataFrame(std_train_data)
    std_valid_data = pd.DataFrame(std_valid_data)

    f = Features({'sent1': Value(dtype='string', id=None),
                  'ce_label': Value(dtype='int8', id=None),
                  'scl_label': Value(dtype='int8', id=None)})

    return {'train': Dataset.from_pandas(std_train_data, features=f),
            'valid': Dataset.from_pandas(std_valid_data, features=f)}


if __name__ == '__main__':
    from types import SimpleNamespace
    from transformers import BertTokenizer

    args = SimpleNamespace(
        path_to_train_data='/home/tmax/kibong_choi/EFL_ContrastiveLearning/data/cs_sharing_30000.csv',
        path_to_valid_data='/home/tmax/kibong_choi/EFL_ContrastiveLearning/data/preprocessed/valid',
        vocab_path='/home/tmax/kibong_choi/EFL_ContrastiveLearning/tokenizer/version_1.9',
        batch_size=4,
        seed=26
    )
    tokenizer = tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                                          do_lower_case=False,
                                                          unk_token='<unk>',
                                                          sep_token='</s>',
                                                          pad_token='<pad>',
                                                          cls_token='<s>',
                                                          mask_token='<mask>',
                                                          model_max_length=256)

    dataloaders = get_std_dataloader(args, tokenizer)
    # dataloaders = get_efl_dataloader(args, tokenizer)

    for batch in dataloaders['train']:
        print(batch['input_ids'].shape)
        print(tokenizer.decode(batch['input_ids'][0]))
        print(batch['scl_label'])
        exit()
