import torch
from torch import nn

EMB_DIM = 2
HIDDEN_SIZE = 3
rnn = nn.GRU(input_size=EMB_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
embedding = nn.Embedding(num_embeddings=7, embedding_dim=EMB_DIM)

input = [torch.LongTensor([1]),
         torch.LongTensor([2,3,4]),
         torch.LongTensor([5,6])]

input_lens = [1, 3, 2]
print(input)

input = nn.utils.rnn.pad_sequence(input, batch_first=True)
print(input)

input = embedding(input)

packed= nn.utils.rnn.pack_padded_sequence(input, batch_first=True, lengths=input_lens, enforce_sorted=False)

outputs, h = rnn(packed)

outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

print('h', h)

print('outputs', outputs)
