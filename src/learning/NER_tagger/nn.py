# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
from utils import shared
from theano.tensor.shared_randomstreams import RandomStreams


class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def __init__(self, input_dim, output_dim, bias=True, activation='sigmoid', name='hidden_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        else:
            raise Exception("Unknown activation function: " % activation)

        # Initialize weights and bias
        self.weights = shared((input_dim, output_dim), name + '__weights')
        self.bias = shared((output_dim, ), name + '__bias')

        # Define parameters
        if self.bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def link(self, input):
        """
        get the output of the Hidden layer.
        The input has to be a tensor with the right most dimension equal to input_dim.(维度匹配)
        :param input:
        :return:
        """
        self.input = input
        self.linear_output = T.dot(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer: word embeddings representations.
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """
    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        """
        :param input_dim: the vocabulary size
        :param output_dim: the embedding dimension
        :param name:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Randomly generate weights
        self.embeddings = shared((input_dim, output_dim), self.name + '__embeddings')

        # Define parameters
        self.params = [self.embeddings]

    def link(self, input):
        """
        Return the embeddings of given indexes.
        Input: tensor of shape (dim*)
        Output: tensor of shape(dim*, output_dim)
        self.embeddins 即是词向量矩阵
        :param input:
        :return:
        """
        self.input = input
        self.output = self.embeddings[self.input]
        return self.output


class DropoutLayer(object):
    """
    Randomly set to 0 values of the input with probability p.
    """
    def __init__(self, p=0.5, name='dropout_layer'):
        """
        p has to be between 0 and 1 (1 excluded).
        p is the probability of dropping out a unit.
        :param p:
        :param name:
        """
        assert 0. <= p < 1.
        self.p = p
        self.rng = RandomStreams(seed=123456)
        self.name = name

    def link(self, input):
        """
        dropout link : apply mask to the input.
        :param input:
        :return:
        """
        if self.p > 0:
            mask = self.rng.binomial(n=1, p=1-self.p, size=input.shape, dtype=theano.config.floatX)
            self.output = input * mask
        else:
            self.output = input

        return self.output


class LSTM(object):
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
         Input: matrix of dimension (sequence_length, input_dim)
         Output: vector of dimension (output_dim)
    With batches:
         Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
         Output: matrix of dimension (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        :param input_dim:
        :param hidden_dim:
        :param with_batch:
        :param name:
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Input gate weights
        self.w_xi = shared((input_dim, hidden_dim), name + '__w_xi')
        self.w_hi = shared((hidden_dim, hidden_dim), name + '__w_hi')
        self.w_ci = shared((hidden_dim, hidden_dim), name + '__w_ci')

        # Output gate weights
        self.w_xo = shared((input_dim, hidden_dim), name + '__w_xo')
        self.w_ho = shared((hidden_dim, hidden_dim), name + '__w_ho')
        self.w_co = shared((hidden_dim, hidden_dim), name + '__w_co')

        # Cell weights
        self.w_xc = shared((input_dim, hidden_dim), name + '__w_xc')
        self.w_hc = shared((hidden_dim, hidden_dim), name + "__w_hc")

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = shared((hidden_dim, ), name + '__b_i')
        self.b_c = shared((hidden_dim, ), name + '__b_c')
        self.b_o = shared((hidden_dim, ), name + '__b_o')

        self.c_0 = shared((hidden_dim, ), name + '__c_0')
        self.h_0 = shared((hidden_dim, ), name + '__h_0')

        # Define parameters
        self.params = [self.w_xi, self.w_hi, self.w_ci, self.w_xo, self.w_ho, self.w_co,
                       self.w_xc, self.w_hc, self.b_i, self.b_c, self.b_o, self.c_0, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.

        The whole sequence is also accessible via self.h, but where self.h of shape
        (sequence_length, batch_size, output_dim)
        :param input:
        :return:
        """
        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) + T.dot(h_tm1, self.w_hi) +
                                 T.dot(c_tm1, self.w_ci) + self.b_i)
            c_t = (1 - i_t) * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) + T.dot(h_tm1, self.w_hc)
                                                   + self.b_c)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) + T.dot(h_tm1, self.w_ho) + self.b_o)
            h_t = o_t * T.tanh(c_t)

            return [c_t, h_t]

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = [for x in [self.c_0, self.h_0]]


















