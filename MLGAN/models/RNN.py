import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
from models import units
import copy


class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)


class Get_embd(nn.Module):
    def __init__(self, options):
        super(Get_embd, self).__init__()
        vocab_size = options['n_diagnosis_codes'] + 1
        model_dim = options['hidden_size']
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.gru = nn.GRU(model_dim, model_dim)  # the gru layer
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = options['dropout_rate']
        self.dropout = nn.Dropout(dropout_rate)
        self.options = options

    def forward(self, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, hidden=None):
        lengths = lengths.tolist()

        temp = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        X = torch.relu(temp)
        packed = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, enforce_sorted=False, batch_first=True)
        # Forward pass through GRU
        gru_out, hidden = self.gru(packed, hidden)  # gru layer
        # Unpack padding
        gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True)  # gru_out: batch_size * seq_len * hidden_size

        output = torch.cat([temp1[idx - 1, :].unsqueeze(1)
                            for idx, temp1 in zip(lengths, gru_out)], dim=1).permute(1, 0)

        return output


class Generator(nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.input_dim_gen = options['hidden_size']
        self.main_gen = nn.Sequential(

            nn.Linear(self.input_dim_gen, self.input_dim_gen),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.input_dim_gen, self.input_dim_gen),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.input_dim_gen, self.input_dim_gen),
            nn.Tanh()
        )

    def forward(self, input):
        temp = self.main_gen(input)
        return temp


class Discriminator(nn.Module):
    def __init__(self, options):
        super(Discriminator, self).__init__()
        self.input_dim_dis = options['hidden_size']
        self.main_dis = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(self.input_dim_dis, self.input_dim_dis),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(self.input_dim_dis, self.input_dim_dis),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(self.input_dim_dis, self.input_dim_dis),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.input_dim_dis, 2)
        )

    def forward(self, input):
        temp = self.main_dis(input)  # batch, 2
        return temp

class NetPredict(nn.Module):
    def __init__(self, options):
        super(NetPredict, self).__init__()
        self.embd_model = Get_embd(options)
        self.discriminator_model = Discriminator(options)

    def forward(self, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, hidden=None):
        embd_representation = self.embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        result = self.discriminator_model(embd_representation)

        return result
