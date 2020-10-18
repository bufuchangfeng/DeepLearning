from torch import nn
import torch


input_size = 3
batch_size = 5
num_layers = 2
hidden_size = 256
num_directions = 2
seq_len = 2


lstm = nn.LSTM(input_size=input_size,
               hidden_size=hidden_size,
               num_layers=num_layers,
               bidirectional=True)

inputs = torch.randn(batch_size, seq_len, input_size)

# print(inputs.shape)  # [5, 2, 3]

inputs = inputs.permute(1, 0, 2)

# print(inputs.shape)  # [2, 5, 3]

hidden = (torch.randn(num_layers*num_directions, batch_size, hidden_size),
          torch.randn(num_layers*num_directions, batch_size, hidden_size))

outputs, hidden = lstm(inputs, hidden)

print(outputs.shape)
# [2, 5, 512]
