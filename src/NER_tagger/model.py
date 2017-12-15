# -*- coding: utf-8 -*-
"""
build the LSTM + CRF models
"""
import torch
from torch.autograd import Variable
import torch.nn as nn

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def to_scalar(var):
    """
    change a Variable a python num
    :param var: Variable, dim=1
    :return: A python num
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    """
    return the argmax as a python int
    :param vec: Variable, (1 x dim)
    :return:
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm.
    x* = max(x1, x2, x3, ..., xn)
    log(exp(x1)+exp(x2)+exp(x3)+...+exp(xn))=x*+ log(exp(x1-x*)+exp(x2-x*)+exp(x3-x*)+...+exp(xn-x*))
    :param vec: 1 x (dim)
    :return:
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # 1 x dim
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def prepare_sequence(sentence, word_to_ix):
    """
    change the sentence to a list of index of words according to the word dictionary.
    :param sentence: a list of words.
    :param word_to_ix: word dictionary
    :return: Variable of a list of word index
    """
    sent_ix = [word_to_ix[w] for w in sentence]
    return Variable(torch.LongTensor(sent_ix))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, char_lstm_dim=25,
                 char_to_ix=None, pre_word_embeds=None, char_embedding_dim=25, use_gpu=False,
                 n_cap=None, cap_embedding_dim=None, use_crf=True, char_mode='CNN'):
        """

        :param vocab_size:
        :param tag_to_ix:
        :param embedding_dim: 词向量维度
        :param hidden_dim:
        :param char_lstm_dim:
        :param char_to_ix:
        :param pre_word_embeds: 是否使用已有的词向量
        :param char_embedding_dim:  字符向量维度
        :param use_gpu: 是否使用gpu
        :param n_cap:
        :param cap_embedding_dim:
        :param use_crf: 是否添加CRF层
        :param char_mode: 字符向量的训练模式(LSTM or CNN)
        """
        super(BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.n_cap = n_cap
        self.cap_embedding_dim = cap_embedding_dim
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_lstm_dim
        self.char_mode = char_mode

        print('char_mode: %s, out_channels: %d, hidden_dim: %d, ' % (char_mode, char_lstm_dim, hidden_dim))

        # cap_embedding???
        if self.n_cap and self.cap_embedding_dim:
            self.cap_embeds = nn.Embedding(self.n_cap, self.cap_embedding_dim)
            init_embedding(self.cap_embeds.weight)

        # 是否使用字符向量？
        if char_embedding_dim is not None:
            self.char_lstm_dim = char_lstm_dim
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(input_size=char_embedding_dim, hidden_size=char_lstm_dim,
                                         num_layers=1, bidirectional=True)
                init_lstm(self.char_lstm)
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                           kernel_size=(3, char_embedding_dim), padding=(2, 0))

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 是否使用训练好的词向量？
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(0.5)
        # 输入到BiLSTM（cap??）
        if self.n_cap and self.cap_embedding_dim:
            if self.char_mode == 'LSTM':
                self.lstm = nn.LSTM(embedding_dim + char_lstm_dim * 2 + cap_embedding_dim, hidden_dim,
                                    bidirectional=True)
            if self.char_mode == 'CNN':
                self.lstm = nn.LSTM(embedding_dim + self.out_channels + cap_embedding_dim, hidden_dim,
                                    bidirectional=True)
        else:
            if self.char_mode == 'LSTM':
                self.lstm = nn.LSTM(embedding_dim + char_lstm_dim * 2, hidden_dim, bidirectional=True)
            if self.char_mode == 'CNN':
                self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)
        init_lstm(self.lstm)

        self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        self.h2_h1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden2tag)
        init_linear(self.hw_gate)
        init_linear(self.hw_trans)

        # 是否使用CRF层
        if self.use_crf:
            # Matrix of transition parameters. Entry i, j is the score of transitioning to i from j.
            self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _score_sentence(self, feats, tags):
        """
        get the score given the tags.
        :param feats: a 2D tensor, len(sentence) * tagset_size, like as
           tags
           end
           o     [- - - ,,,  - ]
           b     [- - - ,,,  - ]
           b-p  [- - - ,,,  - ]
            ...    .....
           L-o   [- - - ,,,  - ]
           start
     steps:   1 2 3 ...   n
        :param tags: ground_truth(正确的标注), a list of ints, length is len(sentence),
              contain START_TAG and STOP_TAG.
        :return:
        """
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])
        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

    def _get_lstm_features(self, sentence, chars2, caps, chars2_length, d):
        """
        输入sentence, 得到lstm_feats
        :param sentence:
        :param chars2:  ?
        :param caps:
        :param chars2_length:  ?
        :param d:
        :return:
        """
        if self.char_mode == 'LSTM':
            # 根据chars2得到字符向量,pack输入lstm
            chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            # pad the lstm_out sequence
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)   # (seq_len, batch, hidden_size)
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index - 1, :self.char_lstm_dim],
                                                  outputs[i, 0, self.char_lstm_dim:]))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                    kernel_size=(chars_cnn_out3.size(2), 1)).view(
                chars_cnn_out3.size(0), self.out_channels)

        embeds = self.word_embeds(sentence)
        if self.n_cap and self.cap_embedding_dim:
            cap_embedding = self.cap_embeds(caps)

        if self.n_cap and self.cap_embedding_dim:
            embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        else:
            embeds = torch.cat((embeds, chars_embeds), 1)

        embeds = embeds.unsqueeze(1)   # (seq_len, batch, input_size)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        """
        计算所有路径之和
        :param feats: a 2D tensor, len(sentence) * tagset_size, like as
           tags
           end
           o     [- - - ,,,  - ]
           b     [- - - ,,,  - ]
           b-p  [- - - ,,,  - ]
            ...    .....
           L-o   [- - - ,,,  - ]
           start
         steps:   1 2 3 ...   n
        :return: alpha, the sum of scores of all paths, compute as
           start - - - - - - - - - -
           1     -    -     -      -
           2    -    -    -      -
           .
           .    -    -     -      -
           .
           n   -    -    -      -
           end -   -    -    -
        """
        init_alphas = torch.FloatTensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = Variable(init_alphas)
        if self.cuda():
            forward_var = forward_var.cuda()   # 1 x tagset_size
        for feat in feats:
            emit_score = feat.view(-1, 1)  # tagset_size x 1
            tag_var = forward_var + self.transitions + emit_score  # tagset_size x tagset_size(broadcast)

            # calculate log_sum_exp
            max_tag_var, _ = torch.max(tag_var, 1)  # 取每行最大值
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), 1)).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        return alpha





