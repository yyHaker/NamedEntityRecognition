import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
batch_size = 2
max_length = 3
hidden_size = 2
n_layers = 1

tensor_in = torch.FloatTensor([[1, 2, 3], [1, 0, 0]]).resize_(2, 3, 1)
tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = [3, 3]  # list of integers holding information about the batch size at each sequence step

# pack it
pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
print(pack)

# initialize
rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))

# forward
out, _ = rnn(pack, h0)
print(out)

# unpack
unpacked = nn_utils.rnn.pad_packed_sequence(out)
print(unpacked)