# -*- coding: utf-8 -*-
import theano
import numpy as np
import theano.tensor as T
ones = theano.shared(np.float32([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
twos = theano.shared(np.float32([[10, 11, 12], [13, 14, 15]]))

print(ones.get_value())

result = T.concatenate([ones, ones], axis=0)  # 在列上连接

print(result.eval())

result = T.concatenate([ones, ones], axis=1)  # 在行上连接

print(result.eval())

# wrong : all the input array dimensions except for the concatenation axis must match exactly
result = T.concatenate([ones, twos], axis=1)
print (result.eval())