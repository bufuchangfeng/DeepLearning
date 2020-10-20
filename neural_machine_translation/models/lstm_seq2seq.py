from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
import _pickle as pickle
import sys
import os
import random


class Encoder(nn.Module):
    def __init__(self, n_words, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_embeddings=n_words, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        self.rnn.flatten_parameters()

        # src [batch_size, seq_len]

        embedded = self.dropout(self.embedding(src))
        # embedded [batch_size, seq_len, emb_dim]

        embedded = embedded.permute(1, 0, 2)
        # embedded [seq_len, batch_size, emb_dim]

        outputs, (h, c) = self.rnn(embedded)
        # outputs [seq_len, batch_size, hidden_dim*1(n_directions)]
        # h (n_layers * 1, batch_size, hid_dim)
        # c (n_layers * 1, batch_size, hid_dim)

        return h, c


class Decoder(nn.Module):
    def __init__(self, n_words, emd_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_words = n_words

        self.embedding = nn.Embedding(num_embeddings=n_words, embedding_dim=emd_dim)
        self.rnn = nn.LSTM(emd_dim, hid_dim, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, n_words)

    def forward(self, words, h, c):
        self.rnn.flatten_parameters()

        # words [batch_size]
        # h c [n_layers*1, batch_size, hid_dim]
        words = words.reshape(1, -1)
        # words[1, batch_size]

        embedded = self.dropout(self.embedding(words))
        # embedded [1, batch_size, emb_dim]

        output, (h, c) = self.rnn(embedded, (h, c))
        # output [1, batch_size, hid_dim]

        output = output.reshape(-1, self.hid_dim)
        # output [batch_size, hid_dim]

        prediction = self.fc(output)
        # prediction [batch_size, n_words]

        return prediction, h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.encoder.to(device)
        self.decoder.to(device)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src [batch_size, src_len]
        # trg [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_n_words = self.decoder.n_words

        outputs = torch.zeros(trg_len, batch_size, trg_n_words).to(self.device)

        h, c = self.encoder(src)

        words = trg[:, 0]
        # words [batch_size]

        for t in range(1, trg_len):
            output, h, c = self.decoder(words, h, c)
            # output [batch_size, n_words]
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            words = trg[:, t] if teacher_force else top1

        return outputs


class CreateDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __getitem__(self, index):
        return torch.LongTensor(self.src[index]), torch.LongTensor(self.trg[index])

    def __len__(self):
        return len(self.src)


def train_and_save():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(n_words=34412, emb_dim=256, hid_dim=256, n_layers=1, dropout=0.2)
    decoder = Decoder(n_words=47147, emd_dim=256, hid_dim=256, n_layers=1, dropout=0.2)
    model = Seq2Seq(encoder, decoder, device)

    torch.distributed.init_process_group(backend="nccl")

    model = model.to(device)

    model = nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    # for test
    # epochs = 1
    epochs = 5
    batch_size = 128

    print('read file zh_data...')

    f = open('./data/zh_data', 'rb+')
    zh_data = pickle.load(f)
    f.close()

    print('read file en_data...')

    f = open('./data/en_data', 'rb+')
    en_data = pickle.load(f)
    f.close()

    # for test
    # train_dataset = CreateDataset(en_data[:8], zh_data[:8])
    # valid_dataset = CreateDataset(en_data[8:16], zh_data[8:16])

    train_dataset = CreateDataset(en_data[:80000], zh_data[:80000])
    valid_dataset = CreateDataset(en_data[80000:100000], zh_data[80000:100000])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    best_valid_loss = sys.maxsize

    print('start training...')

    for epoch in range(epochs):
        model.train()
        for index, (src, trg) in enumerate(train_loader):

            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg)

            output_dim = output.shape[-1]

            output = output.reshape(trg.shape[0], trg.shape[1], -1)

            output = output[:, 1:, :].reshape(-1, output_dim)

            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            loss.backward()

            optimizer.step()

            if index % 10 == 0:
                print('Train Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, loss.item()))

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for index, (src, trg) in enumerate(valid_loader):
                src = src.to(device)
                trg = trg.to(device)

                output = model(src, trg, 0)

                output_dim = output.shape[-1]

                output = output.reshape(trg.shape[0], trg.shape[1], -1)

                output = output[:, 1:, :].reshape(-1, output_dim)

                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)

                valid_loss += loss.item()

            print('Train Epoch [{}/{}], Valid Loss: {:.4f}'
                  .format(epoch + 1, epochs, valid_loss))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                if os.path.exists('./data/seq2seq.pth'):
                    os.remove('./data/seq2seq.pth')
                torch.save(model, './data/seq2seq.pth')
                print('get new model')


def main():
    train_and_save()


if __name__ == '__main__':
    main()
