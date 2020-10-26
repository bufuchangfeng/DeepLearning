import torch
from torch import nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, lens, hidden):

        # x [seq_len, batch_size]
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lens, enforce_sorted=False, batch_first=False)

        _, hidden = self.gru(packed, hidden)

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=True):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(hid_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):

        batch_size = x.shape[0]

        # x [batch_size]
        x = x.reshape(1, -1)
        # x [1, batch_size]

        embedded = self.dropout(self.embedding(x))
        # embedded [1, batch_size, emb_dim]
        #          [seq_len, batch_size, emb_dim]

        outputs, hidden = self.gru(embedded, hidden)
        # outputs []

        outputs = outputs.reshape(batch_size, -1)

        outputs = self.fc(outputs)

        return outputs, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_trg_len=100):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.encoder.to(device)
        self.decoder.to(device)

        self.encoder_directions = 2 if encoder.bidirectional else 1

        self.max_trg_len = max_trg_len

    def forward(self, src, src_lens, trg=None, trg_lens=None, teacher_forcing_ratio=0.5):

        # src [seq_len, batch_size]
        # trg [seq_len, batch_size]
        batch_size = src.shape[1]

        hidden = torch.zeros(self.encoder.n_layers*self.encoder_directions, batch_size,
                             self.encoder.hid_dim, device=self.device)

        hidden = self.encoder(src, src_lens, hidden)

        x = trg[0, :]

        max_trg_len = max(trg_lens)

        outputs = torch.zeros(max_trg_len, batch_size, self.decoder.output_dim)

        for t in range(1, max_trg_len):
            teacher_focing = random.random() < teacher_forcing_ratio

            output, hidden = self.decoder(x.to(self.device), hidden)

            outputs[t] = output

            top1 = output.argmax(dim=1)

            x = trg[t, :] if teacher_focing else top1

        return outputs

    def translate(self, src, src_lens, trg):
        # src [seq_len, batch_size]
        batch_size = src.shape[1]

        hidden = torch.zeros(self.encoder.n_layers*self.encoder_directions, batch_size,
                             self.encoder.hid_dim, device=self.device)

        hidden = self.encoder(src, src_lens, hidden)

        x = trg[0, :]

        outputs = torch.zeros(self.max_trg_len, batch_size, self.decoder.output_dim)

        for t in range(1, self.max_trg_len):

            output, hidden = self.decoder(x.to(self.device), hidden)

            outputs[t] = output

            top1 = output.argmax(dim=1)

            x = top1

        return outputs
