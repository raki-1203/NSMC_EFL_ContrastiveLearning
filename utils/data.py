import os
import sys
import pickle
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from datasets import Features, Value, Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from utils.label_descriptions import efl_sentiment_label_descriptions, sentiment_scl_label_table, \
    std_sentiment_label_table


def get_efl_dataloader(args, tokenizer, mecab):
    dataset = _get_efl_nsmc_dataset(args)

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

    nsmc_dataloader_dict = {'train': DataLoader(train_dataset,
                                                collate_fn=data_collator,
                                                shuffle=True,
                                                batch_size=args.batch_size),
                            'valid': DataLoader(valid_dataset,
                                                collate_fn=data_collator,
                                                shuffle=False,
                                                batch_size=args.batch_size)}

    return nsmc_dataloader_dict


def _get_efl_nsmc_dataset(args):
    if not os.path.exists(os.path.join(project_dir, 'data', args.path_to_train_data)):
        train_data = {}
        valid_data = {}
        test_data = {}

        train_df = pd.read_csv(os.path.join(project_dir, 'data/ratings_train.txt'), sep='\t')
        train_df.dropna(inplace=True)
        test_df = pd.read_csv(os.path.join(project_dir, 'data/ratings_test.txt'), sep='\t')
        test_df.dropna(inplace=True)

        train_df, valid_df = train_test_split(train_df, test_size=0.3, shuffle=True, stratify=train_df['label'],
                                              random_state=args.seed)

        train_positive_mask = train_df['label'] != 0
        train_negative_mask = train_df['label'] == 0
        valid_positive_mask = valid_df['label'] != 0
        valid_negative_mask = valid_df['label'] == 0
        test_positive_mask = test_df['label'] != 0
        test_negative_mask = test_df['label'] == 0

        train_data['negative'] = train_df.loc[train_negative_mask]['document'].values.tolist()
        train_data['positive'] = train_df.loc[train_positive_mask]['document'].values.tolist()

        valid_data['negative'] = valid_df.loc[valid_negative_mask]['document'].values.tolist()
        valid_data['positive'] = valid_df.loc[valid_positive_mask]['document'].values.tolist()

        test_data['negative'] = test_df.loc[test_positive_mask]['document'].values.tolist()
        test_data['positive'] = test_df.loc[test_negative_mask]['document'].values.tolist()

        efl_train_data = []
        efl_valid_data = []
        efl_test_data = []

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

        for true_category in test_data.keys():
            for sent in test_data[true_category]:
                for category, label_description in efl_sentiment_label_descriptions.items():
                    new_example = {}
                    new_example['sent1'] = sent
                    new_example['sent2'] = label_description

                    new_example['ce_label'] = std_sentiment_label_table[true_category]
                    new_example['scl_label'] = sentiment_scl_label_table[true_category][category]
                    efl_test_data.append(new_example)

        efl_train_data = pd.DataFrame(efl_train_data)
        efl_valid_data = pd.DataFrame(efl_valid_data)
        efl_test_data = pd.DataFrame(efl_test_data)

        f = Features({'sent1': Value(dtype='string', id=None),
                      'sent2': Value(dtype='string', id=None),
                      'ce_label': Value(dtype='int8', id=None),
                      'scl_label': Value(dtype='int8', id=None)})

        datasets = DatasetDict({'train': Dataset.from_pandas(efl_train_data, features=f),
                                'valid': Dataset.from_pandas(efl_valid_data, features=f),
                                'test': Dataset.from_pandas(efl_test_data, features=f),
                                })

        datasets.save_to_disk(os.path.join(project_dir, 'data', args.path_to_train_data))
    else:
        datasets = load_from_disk(os.path.join(project_dir, 'data', args.path_to_train_data))

    return datasets
