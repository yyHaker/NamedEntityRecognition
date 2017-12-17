# -*- coding: utf-8 -*-
"""
     theano.tensor.alloc(value,*shape):生成一个变化的tensor，维度是shape大小的，
但是值但是由value填充。
"""
import numpy as np
import theano
import theano.tensor as T


X = T.matrix()
e = T.alloc(1, 4, 3)
p = theano.function([X], e + X)
a = np.random.rand(4, 3).astype('float32')
print a
print p(a)
