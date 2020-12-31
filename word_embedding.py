"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from fc import FCNet
import torch.nn.functional as F

#word embedding :
#1. use nn.Embedding
#2. use nn.Dropout layer delete some note
#3. design some args such as ntoken word's number,emb_dim word's dim
#4. design some meths such as init_embedding, but forward function must need.

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.

    """
    def __init__(self, ntoken, emb_dim, dropout, op=''):
        '''
        :param ntoken: 19901
        :param emb_dim: 300
        :param dropout: 0.0
        :param op: ''
        '''
        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        #if true ,emb's weight not grad
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False  # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim
    #load glove6b_init_300d.npy file init embedding's weight
    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        '''
        :param np_file: glove6b_init_300d.npy
        :param tfidf:None
        :param tfidf_weights: None
        :return:
        '''
        #[19901,300]
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init,
                                         torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init)  # (N x N') x (N', F)
            if 'c' in self.op:
                self.emb_.weight.requires_grad = True
        if 'c' in self.op:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        # x[64,14] tensor([[    0,     1,    11,   331,  1185,    36,     0,     0,     0,     0,
        #emb[64,14,300]
        emb = self.emb(x)
        # emb = emb.type(torch.LongTensor)
        if 'c' in self.op:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout,
                 rnn_type='GRU'):
        """Module for question embedding
        in_dim:300
        num_hid:1024
        nlayers:1
        bidirect:False
        dropout:0.0
        rnn_type='GRU'
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU \
            if rnn_type == 'GRU' else None
        #(300,1024,1)
        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        #weight = next(self.parameters()).data
        weight = 0
        weight = torch.tensor(weight,dtype=torch.float32)
        weight = weight.cuda()#tensor type 0 and it use cuda
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid) #(1,512,1024)
        if self.rnn_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(),
                    weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim] [64,14,300]
        batch = x.size(0)
        hidden = self.init_hidden(batch) #[1,64,1024]
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden) #output:torch.Size([64, 14, 1024])

        if self.ndirections == 1:
            return output[:, -1] #[64,1024] 选取的是14个长度的最后一个数据

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


#问题自注意力模型
class QuestionSelfAttention(nn.Module):
    def __init__(self, num_hid, dropout):
        super(QuestionSelfAttention, self).__init__()
        self.num_hid = num_hid
        self.drop = nn.Dropout(dropout)
        # W1_self_att_q：Dropout(p=0.2, inplace=False)
        self.W1_self_att_q = FCNet(dims=[num_hid, num_hid], dropout=dropout,act=None)
        #
        self.W2_self_att_q = FCNet(dims=[num_hid, 1], act=None)

    def forward(self, ques_feat):
        '''
        ques_feat: [batch, 14, num_hid]
        '''
        batch_size = ques_feat.shape[0]
        q_len = ques_feat.shape[1]

        # (batch*14,num_hid)
        ques_feat_reshape = ques_feat.contiguous().view(-1, self.num_hid)
        # (batch, 14)
        atten_1 = self.W1_self_att_q(ques_feat_reshape)
        atten_1 = torch.tanh(atten_1)
        atten = self.W2_self_att_q(atten_1).view(batch_size, q_len)
        # (batch, 1, 14)
        weight = F.softmax(atten.t(), dim=1).view(-1, 1, q_len)
        ques_feat_self_att = torch.bmm(weight, ques_feat)
        ques_feat_self_att = ques_feat_self_att.view(-1, self.num_hid)
        # (batch, num_hid)
        ques_feat_self_att = self.drop(ques_feat_self_att)
        return ques_feat_self_att
