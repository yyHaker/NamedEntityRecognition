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

    def _forward(self, obs_seq):
        """
        观测序列概率的前向算法
        :param obs_seq:
        :return:
        """
        N = self.A.shape[0]
        T = len(obs_seq)

        alpha = np.zeros((N, T))

        alpha[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for i in range(N):
                alpha[i, t] = np.dot(alpha[:, t-1], self.A[:, i]) * self.B[i, obs_seq[t]]
        return alpha

    def _backward(self, obs_seq):
        """
        观测序列的后向算法
        :param obs_seq:
        :return:
        """
        N = self.A.shape[0]
        T = len(obs_seq)

        beta = np.zeros((N, T))

        beta[:, -1:] = 1
        for t in reversed(range(T-1)):
            for i in range(N):
                beta[i, t] = np.sum(beta[:, t+1] * self.A[i, :] * self.B[:, obs_seq[t+1]])

        return beta

    def viterbi(self, obs_seq):
        """
        viterbi algorithm.
        :param obs_seq: observation sequence
        :return: V : numpy.ndarray, V[s][t]=Maximum probability of an observation sequence ending
        at time  't' with final state 's'
                   prev: numpy.ndarray, Contains a pointer to the previous state at t-1 that maximizes
        V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)

        prev = np.zeros((T-1, N), dtype=int)
        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))

        V[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for i in range(N):
                seq_probs = V[:, t-1] * self.A[:, i] * self.B[i, obs_seq[t]]
                prev[t-1, i] = np.argmax(seq_probs)
                V[i, t] = np.max(seq_probs)
        return V, prev

    def build_viterbi_path(self, prev, last_state):
        """
        Return a state path ending in last_state in reverse order.
        :param prev:
        :param last_state:
        :return:
        """
        T = len(prev)
        yield(last_state)
        for i in range(T-1, -1, -1):
            yield (prev[i, last_state])
            last_state = prev[i, last_state]

    def state_path(self, obs_seq):
        """
        get the probability of the optimal state path and the optimal state
        path for the observation sequence
        :param obs_seq:
        :return: V[last_state, -1]: float, probability of the optimal state path
                     path: list(int), optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        # build state path with greatest probability
        last_state = np.argmax(V[:, -1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state, -1], reversed(path)









