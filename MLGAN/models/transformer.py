import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
from torch.autograd import Variable
import random
import time
import shutil

class Generator(nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.input_dim_gen = options['hidden_size']
        self.main_gen = nn.Sequential(

            nn.Linear(self.input_dim_gen*2, self.input_dim_gen),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.input_dim_gen, self.input_dim_gen),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.input_dim_gen, self.input_dim_gen*2),
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
            nn.Linear(self.input_dim_dis*2, self.input_dim_dis),
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
        temp = self.main_dis(input)    # batch, 2
        return temp

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

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention

class PositionalEncoding(nn.Module):

    def __init__(self, options, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        self.options = options
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = np.zeros([1, d_model])
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        position_encoding = np.concatenate((pad_row, position_encoding), axis=0)
        # position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding_weight = position_encoding


        # self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        # self.position_encoding.weight = nn.Parameter(position_encoding,
        #                                              requires_grad=True)


    def forward(self, input_len):
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind

        position_encoding = nn.Embedding(self.max_seq_len + 1, self.d_model)
        new_position_encoding = torch.from_numpy(self.position_encoding_weight.astype(np.float32))
        position_encoding.weight = nn.Parameter(new_position_encoding, requires_grad=False)

        input_pos = torch.LongTensor(pos)
        temp = position_encoding(input_pos).to(self.options['device'])

        return temp

def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

def ccompute_ind_pos(input_len):
    max_len = torch.max(input_len)
    tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

    pos = np.zeros([len(input_len), max_len])
    for ind, length in enumerate(input_len):
        for pos_ind in range(1, length + 1):
            pos[ind, pos_ind - 1] = pos_ind

    input_pos = torch.LongTensor(pos)
    input_pos = input_pos.to(input_len.device)

    return input_pos


class EncoderNew(nn.Module):
    def __init__(self,
                 options,
                 vocab_size,
                 max_seq_len,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 dropout=0.0):
        super(EncoderNew, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        # self.weight_layer = torch.nn.Linear(model_dim, 1)
        self.pos_embedding = PositionalEncoding(options, model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):

        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output += time_feature
        ind_pos = ccompute_ind_pos(input_len.unsqueeze(1))
        output_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos.detach()
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)

        return output

class TimeEncoder(nn.Module):
    def __init__(self):
        super(TimeEncoder, self).__init__()
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(64, 64)

    def forward(self, seq_time_step, final_queries, mask):
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)

        return torch.softmax(selection_feature, 1)

class Get_embd(nn.Module):
    def __init__(self, options):
        super(Get_embd, self).__init__()
        self.options = options
        self.feature_encoder = EncoderNew(options, options['n_diagnosis_codes'] + 1, 300, num_layers=1)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.time_encoder = TimeEncoder()
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        self.self_layer = torch.nn.Linear(256, 1)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths):

        features = self.feature_encoder(diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)

        return averaged_features

class NetPredict(nn.Module):
    def __init__(self, options):
        super(NetPredict, self).__init__()
        self.embd_model = Get_embd(options)
        self.discriminator_model = Discriminator(options)

    def forward(self, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, hidden=None):
        embd_representation = self.embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        result = self.discriminator_model(embd_representation)

        return result