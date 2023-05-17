import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
from models import units


class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class Get_embd(nn.Module):
    def __init__(self, options):
        super(Get_embd, self).__init__()
        self.num_code = options['n_diagnosis_codes'] + 1
        self.nb_head = 3
        self.temp_hidden = options['visit_size'] * self.nb_head
        self.hidden_size = options['hidden_size']
        self.pre_embedding = Embedding(self.num_code, self.hidden_size)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(self.temp_hidden))
        bound = 1 / math.sqrt(self.num_code)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.WQ = nn.Linear(self.hidden_size, self.temp_hidden)
        self.WK = nn.Linear(self.hidden_size, self.temp_hidden)
        self.WV = nn.Linear(self.hidden_size, self.temp_hidden)

        self.softmax = nn.Softmax(dim=-1)
        self.bigru = nn.GRU(self.temp_hidden, self.hidden_size, batch_first=True, bidirectional=True)
        self.Ws = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.tanh = nn.Tanh()
        self.Us = nn.Linear(self.hidden_size, 1)

    def forward(self, diagnosis_codes, seq_time_step, mask_batch, mask_final, mask_code, lengths):
        mask_code = mask_code.bool()
        mask_batch = mask_batch.squeeze(2)
        cur_visit = self.pre_embedding(diagnosis_codes) * mask_code    # batch, seq_len, max_code_cur_batch, hidden_size
        batch_size, seq_len, max_code_cur_batch, hidden_size = cur_visit.size()
        query = self.WQ(cur_visit).view(batch_size * self.nb_head, seq_len, max_code_cur_batch, -1)
        key = self.WK(cur_visit).view(batch_size * self.nb_head, seq_len, max_code_cur_batch, -1)
        value = self.WV(cur_visit).view(batch_size * self.nb_head, seq_len, max_code_cur_batch, -1)
        # batch_size * self.nb_head, seq_len, max_code_cur_batch, 64
        scale = (key.size(-1) // self.nb_head) ** -0.5

        # compute attention
        key = key.transpose(2, 3)
        attention = torch.matmul(query, key)      # batch_size, self.nb_head, seq_len, max_code_cur_batch, max_code_cur_batch
        attention = attention * scale
        mask_code = mask_code.repeat(self.nb_head, 1, 1, 1).permute(0, 1, 3, 2)
        attention = attention.masked_fill_(~mask_code, -np.inf)
        mask_code_new = mask_code.clone()
        mask_code_new = mask_code_new.any(dim=-1).unsqueeze(-1)
        attention = attention.masked_fill_(~mask_code_new, 1)
        attention = self.softmax(attention)   # batch * 3, seq_len, max_code_cur_batch, max_code_cur_batch

        mask_code = mask_code.permute(0, 1, 3, 2)
        temp_output = torch.matmul(attention, value)    # batch * 3, seq_len, max_code_cur_batch, 64
        float_mask_code = mask_code.type(temp_output.type())
        temp_output = temp_output * float_mask_code

        nominator = torch.sum(temp_output, dim=2)
        denominator = torch.sum(mask_code.type(nominator.type()), dim=2)

        mask_code_new = mask_code_new.squeeze(-1)
        denominator = denominator.masked_fill_(~mask_code_new, 10)
        temp = nominator / denominator
        temp = temp.reshape(batch_size, seq_len, -1)

        output, hn = self.bigru(temp)
        u = self.tanh(self.Ws(output))
        us = self.Us(u).squeeze(dim=-1)
        alpha = us.masked_fill_(mask_batch, -np.inf)
        alpha = self.softmax(alpha)
        h = torch.einsum('bs,bsd->bd', [alpha, output])

        return h

class Discriminator(nn.Module):
    def __init__(self, options):
        super(Discriminator, self).__init__()
        self.input_dim = options['hidden_size'] * 2
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.25),

            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.input_dim, 2)
        )

    def forward(self, input):
        temp = self.main(input)    # batch, 2
        return temp

class Generator(nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.input_dim = options['hidden_size'] * 2
        self.main = nn.Sequential(

            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.input_dim, self.input_dim),
            nn.Tanh()
        )

    def forward(self, input):
        temp = self.main(input)
        return temp