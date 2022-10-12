import os

import torch

from torch import nn
from transformers import AutoConfig, RobertaModel, AutoModel


class EFLContrastiveLearningModel(nn.Module):
    def __init__(self, args):
        super(EFLContrastiveLearningModel, self).__init__()
        self.args = args

        if 'efl' in args.method:
            num_labels = 2
        else:
            if args.task == 'sentiment':
                num_labels = 2
            elif args.task == 'category':
                num_labels = 3
            else:
                raise NotImplementedError

        self.model_config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

        if os.path.isdir(args.model_name_or_path):
            self.encoder = RobertaModel.from_pretrained(args.model_name_or_path)
        else:
            self.encoder = AutoModel.from_pretrained(args.model_name_or_path)

        self.classifier = ClassificationHead(self.model_config)

        if args.pooler_option == 'cls':
            pass
        elif args.pooler_option == 'base':
            pass
        elif args.pooler_option == 'twolayer':
            self.two_layer_pooler = TwoLayerPooler(self.encoder.config)
        else:
            raise Exception('pooler_option must be one of (cls, base, twolayer)')

    def forward(self, **batch):
        """
        Returns:
            {
                logits : logit for CE (BS, NUM_LABELS)
                embeddings : embedding for SCL (BS, HIDDEN)
            }
        """

        output = self.encoder(**batch)

        # last_hidden_state.shape : (BS, SEQ_LEN, HIDDEN)
        last_hidden_state = output.last_hidden_state

        logits = self.classifier(last_hidden_state)

        if self.args.pooler_option == 'cls':
            embeddings = last_hidden_state[:, 0, :]

        elif self.args.pooler_option == 'base':
            embeddings = output.pooler_output

        elif self.args.pooler_option == 'twolayer':
            embeddings = self.two_layer_pooler(last_hidden_state[:, 0, :])

        return {
            'logits': logits,
            'embeddings': embeddings
        }


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(ClassificationHead, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TwoLayerPooler(nn.Module):
    """ two dense + batch norm pooler """

    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine = False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)
