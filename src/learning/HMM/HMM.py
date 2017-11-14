# -*- coding: utf-8 -*-
"""
实现隐马尔科夫模型算法，基于wiki上的一个经典的例子。

参考博客：http://www.hankcs.com/ml/hidden-markov-model.html
"""
import numpy as np


class HMM(object):
    def __init__(self, A, B, pi):
        """
        order 1 Hidden Markov model, lambda = (A, B, pi)
        :param A: numpy.array, state transition probability matrix
        :param B: numpy.array, output emission probability matrix with shape(N, M),
                          N is the number of state, M is the number of observations.
        :param pi: numpy.array, initial state probability vector
        """
        self.A = A
        self.B = B
        self.pi = pi

    def simulate(self, T):
        """
        generate the observation sequence, given the lambda(A, B, pi)
        :param T:
        :return:
        """
        # 接受一个概率分布，然后生成改分布下的一个样本
        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        # 按照初始状态分布生成第一个状态
        states[0] = draw_from(self.pi)
        # 取出状态对应的观测的概率分布，生成一个观测
        observations[0] = draw_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t-1], :])
            observations[t] = draw_from(self.B[states[t], :])
        return observations, states


