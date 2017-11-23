# -*- coding: utf-8 -*-
import theano
import numpy as np


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    :param shape:
    :param name:
    :return:
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)
