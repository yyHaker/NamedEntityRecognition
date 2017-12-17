# -*- coding: utf-8 -*-
import torch
import numpy as np
from  torch.autograd import Variable
# convert numpy to tensor or vise versa
np_data = np.array([[2, 3, 4], [3, 6, 5]])
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array', np_data,
    '\ntorch_tensor', torch_data,
    '\ntensor to array', tensor2array
)
