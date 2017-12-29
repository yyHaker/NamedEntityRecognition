# -*- coding: utf-8 -*-
import torch
import numpy as np

a = np.array([1, 2, 3])   # 默认为int32
t = torch.from_numpy(a)
print(t)
# [torch.IntTensor of size 3]

t = t.view(1, -1)
print(t)

a = a.astype(dtype='int64')
t = torch.from_numpy(a)
print(t)
# [torch.LongTensor of size 3]

f = torch.LongTensor([[1]]*5)
print(f)
"""
torch.transpose()
"""
a = np.array([
    [[1, 2, 3, 4],
    [5, 6, 7, 8]],
    [[6, 7, 7, 8],
     [4, 5, 6, 7]],
    [[6, 7, 7, 5],
     [9, 5, 6, 4]]
])
a = torch.from_numpy(a)
print(a)  # 3 x 2 x 4
a = a.transpose(0, 1)
print(a)  # 2 x 3 x 4
print("-----------------------------------------")
for i, v in enumerate(a):
    print(i, v)

