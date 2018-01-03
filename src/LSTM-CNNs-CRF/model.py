# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

from const import *


def log_sum_exp(input, keepdim=False):
    """
    对input中的每一行计算log_sum_exp
    :param input:
    :param keepdim:
    :return:
    """
    assert input.dim() == 2
    max_scores, _ = input.max(dim=-1, keepdim=True)  # 每一行的最大值
    output = input - max_scores.expand_as(input)
    return max_scores + torch.log(torch.sum(torch.exp(output), dim=-1, keepdim=keepdim))


def gather_index(input, index):
    """
    按照index来收集input中的值.
    (封装了在batch_size x tag_size 中找寻batch_size个tag所以对应的分数)
    :param input: 2个维度
    :param index: 1个维度
    :return:
    """
    assert input.dim() == 2 and index.dim() == 1
    index = index.unsqueeze(1).expand_as(input)
    output = torch.gather(input, 1, index)
    return output[:, 0]


class CRF(nn.Module):
    def __init__(self, label_size, is_cuda):
        """

        :param label_size: label size
        :param is_cuda: use cuda or not
        """
        super().__init__()
        self.label_size = label_size  # 是否包含START和STOP标签？
        # transition[i][j]表示从标签 j -> i的转换分数
        self.transitions = nn.Parameter(
            torch.randn(label_size, label_size))
        self._init_weight()
        self.torch = torch.cuda if is_cuda else torch

    def _init_weight(self):
        """初始化transitions矩阵"""
        init.xavier_uniform(self.transitions)
        # 任何标签不可能->START, STOP不能->任何标签
        self.transitions.data[START, :].fill_(-10000.)
        self.transitions.data[:, STOP].fill_(-10000.)

    def _score_sentence(self, input, tags):
        """
        批量的求得每个句子的分数.
        :param input: 每个句子的每个词被标记为相应tag的分数， (batch, seq, label_size),
           不包括START和STOP
        :param tags: 该句子对应的真实标签， (batch, seq_labels), 不包括START和STOP
        :return:
              每个seq的分数，shape(batch_size)
        """
        bsz, sent_len, l_size = input.size()
        score = Variable(self.torch.FloatTensor(bsz).fill_(0.))  # batch_size, 统计每个seq的分数

        s_score = Variable(self.torch.LongTensor([[START]]*bsz))  # batch_size x 1

        tags = torch.cat([s_score, tags], dim=-1)  # 在列上连接 batch_size x (1 + tag_size)，tags索引表
        input_t = input.transpose(0, 1)  # (sent_length, batch_size, label_size)
        # 对seq中的每个标签依次计算分数
        # tags[:, i] 表示第i个位置的标签索引(START默认索引为1)
        for i, words in enumerate(input_t):
            temp = self.transitions.index_select(1, tags[:, i])  # 挑选这么多列, tag_size x batch_size
            # 计算i -> i+1 的分数
            bsz_t = gather_index(temp.transpose(0, 1), tags[:, i + 1])  # (batch_size x tag_size), (batch_size,)
            # 计算在i+1标签分数
            w_step_score = gather_index(words, tags[:, i+1])  # words(batch_size, label_size)
            score = score + bsz_t + w_step_score

        temp = self.transitions.index_select(1, tags[:, -1])   # STOP的前一个位置
        # 计算转移到STOP标签的分数
        bsz_t = gather_index(temp.transpose(0, 1),
                    Variable(self.torch.LongTensor([STOP]*bsz)))
        return score+bsz_t

    def forward(self, input):
        """
        计算所有路径的分数(公式中的分母)
        :param input: 每个句子的每个词被标记为相应tag的分数， (batch, seq, label_size)
        :return:
        """
        bsz, sent_len, l_size = input.size()
        # init_alphas:
        init_alphas = self.torch.FloatTensor(bsz, self.label_size).fill_(-10000.)
        init_alphas[:, START].fill_(0.)
        # forward_var，shape(bsz, label_size)，
        forward_var = Variable(init_alphas)

        input_t = input.transpose(0, 1)  # (seq, batch, label_size)
        for words in input_t:
            alphas_t = []  # 到第t步的分数(发射分数和转移分数) , (batch, label_size)
            for next_tag in range(self.label_size):
                emit_score = words[:, next_tag].contiguous()  # (batch, )
                emit_score = emit_score.unsqueeze(1).expand_as(words)  # (batch, 1) -> (batch, label_size)
                # 转移分数 (1, label_size) -> (batch, label_size)
                trans_score = self.transitions[next_tag, :].view(1, -1).expand_as(words)
                # 到next_tag的发射分数和转移分数(按batch计算)
                next_tag_var = forward_var + trans_score + emit_score   # (batch, label_size)
                alphas_t.append(log_sum_exp(next_tag_var, True))   # (batch, 1)
            forward_var = torch.cat(alphas_t, dim=-1)  # 在列上连接

        return log_sum_exp(forward_var)

    def viterbi_decode(self, input):
        """
        找到最好的标注序列
        :param input: (batch, seq, label_size)
        :return:
        """
        backpointers = []
        bsz, sent_len, l_size = input.size()

        init_vvars = self.torch.FloatTensor(bsz, self.label_size).fill_(-10000.)
        init_vvars[:, START].fill_(0.)
        forward_var = Variable(init_vvars)

        input_t = input.transpose(0, 1)  # (seq, batch, label_size)
        for words in input_t:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.label_size):
                _trans = self.transitions[next_tag].view(1, -1).expand_as(words)
                next_tag_var = forward_var + _trans
                best_tag_scores, best_tag_ids = torch.max(next_tag_var, 1, keepdim=True)  # bsz
                # save the best path and scores
                bptrs_t.append(best_tag_ids)
                viterbivars_t.append(best_tag_scores)

            forward_var = torch.cat(viterbivars_t, -1) + words
            backpointers.append(torch.cat(bptrs_t, dim=-1))

        terminal_var = forward_var + self.transitions[STOP].view(1, -1).expand(bsz, l_size)
        _, best_tag_ids = torch.max(terminal_var, 1)

        best_path = [best_tag_ids]
        for bptrs_t in reversed(backpointers):
            best_tag_ids = gather_index(bptrs_t, best_tag_ids)  # bsz
            best_path.append(best_tag_ids.contiguous().view(-1, 1))  # bsz x 1

        best_path.pop()
        best_path.reverse()

        return torch.cat(best_path, dim=-1)  # 在列上拼接


class BiLSTM(nn.Module):
    def __init__(self, word_size, word_ebd_dim, kernel_num, lstm_hsz, lstm_layers, dropout, batch_size):
        """

        :param word_size: 词集大小
        :param word_ebd_dim: 词向量维度
        :param kernel_num: (字符向量维度)
        :param lstm_hsz: lstm hidden size
        :param lstm_layers: lstm layers
        :param dropout: dropout rate
        :param batch_size: batch size
        """
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hsz = lstm_hsz
        self.batch_size = batch_size

        self.word_ebd = nn.Embedding(word_size, word_ebd_dim)
        self.lstm = nn.LSTM(input_size=word_ebd_dim+kernel_num,
                            hidden_size=lstm_hsz // 2,
                            num_layers=lstm_layers,
                            batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                            dropout=dropout,   # rnn 层与层之间使用dropout
                            bidirectional=True)
        self._init_weights()

    def _init_weights(self, scope=1.):
        """使用均匀分布初始化 词向量"""
        self.word_ebd.weight.data.uniform_(-scope, scope)

    def forward(self, words, char_feats, hidden=None):
        """

        :param words: (batch, seq)
        :param char_feats: (batch, seq)
        :param hidden:
        :return:
                'output':  (batch, seq, hidden_size)
                'hidden': (h_n, c_n)
        """
        encode = self.word_ebd(words)   # (batch, seq, feature)
        encode = torch.cat((char_feats, encode), dim=-1)  # 在列上连接, (batch, seq, feature)
        output, hidden = self.lstm(encode, hidden)
        return output, hidden

    def init_hidden(self):
        """
        初始化LSTM的hidden
        :return:
        """
        weight = next(self.parameters()).data
        # new（）怎么理解？
        return (Variable(weight.new(self.lstm_layers*2, self.batch_size, self.lstm_hsz//2).zero_()),
            Variable(weight.new(self.lstm_layers*2, self.batch_size, self.lstm_hsz//2).zero_()))


class CNN(nn.Module):
    def __init__(self, char_size, char_ebd_dim, kernel_num, filter_size, dropout):
        """

        :param char_size: 字符集大小
        :param char_ebd_dim: 字符向量维度
        :param kernel_num: CNN 输出channels
        :param filter_size: CNN kernel_size的1维大小
        :param dropout: float, dropout rate
        """
        super().__init__()

        self.char_size = char_size
        self.char_ebd_dim = char_ebd_dim
        self.kernel_num = kernel_num
        self.filter_size = filter_size
        self.dropout = dropout

        self.char_ebd = nn.Embedding(self.char_size, self.char_ebd_dim)
        self.char_cnn = nn.Conv2d(in_channels=1,
                                  out_channels=self.kernel_num,
                                  kernel_size=(self.filter_size, self.char_ebd_dim))
        self._init_weight()

    def _init_weight(self, scope=1.):
        """
        使用均匀分布初始化char_ebd
        :param scope:
        :return:
        """
        init.xavier_uniform(self.char_ebd.weight)

    def forward(self, input):
        """
        extract character-level representation of a given word.
        :param input: (batch_size, word_len, char_len)
        :return:
        """
        bsz, word_len, char_len = input.size()
        encode = input.view(-1, char_len)
        # (batch_size x word_len , char_len, char_ebd_dim)  -> (batch_size x word_len , 1,  char_len, char_ebd_dim)
        encode = self.char_ebd(encode).unsqueeze(1)
        encode = F.relu(self.char_cnn(encode))   # (batch_size x word_len, out_channels, ?, char_ebd_dim)
        encode = F.max_pool2d(encode, kernel_size=(encode.size(2), 1))  # (batch_size x word_len, out_channels, 1, char_ebd_dim)
        encode = F.dropout(encode.squeeze(), p=self.dropout)  # (batch_size x word_len, out_channels, char_ebd_dim)
        return encode.view(bsz, word_len, -1)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.cnn = CNN(self.char_size, self.char_ebd_dim,
            self.kernel_num, self.filter_size, self.dropout)
        self.bilstm = BiLSTM(self.word_size, self.word_ebd_dim, self.kernel_num,
                    self.lstm_hsz, self.lstm_layers, self.dropout, self.batch_size)

        self.logistic = nn.Linear(self.lstm_hsz, self.label_size)
        self.crf = CRF(self.label_size, self.use_cuda)
        self._init_weights()

    def forward(self, words, chars, labels, hidden=None):
        """

        :param words: (batch, seq)
        :param chars: (batch, seq)
        :param labels: (batch, seq_labels)
        :param hidden:
        :return:
        """
        char_feats = self.cnn(chars)
        output, _ = self.bilstm(words, char_feats, hidden)  # (batch, seq, hidden_size)
        output = self.logistic(output)  # (batch, seq,  label_size)
        pre_score = self.crf(output)
        label_score = self.crf._score_sentence(output, labels)
        return (pre_score-label_score).mean(), None

    def predict(self, word, char):
        char_out = self.cnn(char)
        lstm_out, _ = self.bilstm(word, char_out)
        out = self.logistic(lstm_out)
        return self.crf.viterbi_decode(out)

    def _init_weights(self, scope=1.):
        """
        初始化线性层的weight和bias
        :param scope:
        :return:
        """
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

# TODO: 1. load pre_trained Embeddings ?
# TODO: 2. how to test the model and calculate the F1 score ?
# TODO: 3. batch training can speed up or even improve the results ?

