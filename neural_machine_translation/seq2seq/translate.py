import json
import numpy as np
import os
import sys
import torch
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from seq2seq.model import *
from seq2seq.train import tokenize_en


with open('zh_idx2word', 'r', encoding='utf-8') as f:
    zh_idx2word = json.load(f)

with open('en_word2idx', 'r') as f:
    en_word2idx = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder(len(en_word2idx), emb_dim=256, hid_dim=512, n_layers=4, dropout=0.5, bidirectional=True)
decoder = Decoder(len(zh_idx2word), emb_dim=256, hid_dim=512, n_layers=4, dropout=0.5, bidirectional=True)

model = Seq2Seq(encoder, decoder, device, 100)
model.load_state_dict(torch.load('model.pth'))

model.eval()
with torch.no_grad():
    while True:
        en = input('please input an English sentence：')
        en = tokenize_en(en)
        en.append('<eos>')

        en_len = len(en)

        en_data = []

        for w in en:
            if w in en_word2idx.keys():
                en_data.append(en_word2idx[w])
            else:
                en_data.append(en_word2idx['<unk>'])
        en = en_data
        en = [en]

        en = torch.LongTensor(en)

        output = model.translate(en.to(device), [en_len], torch.LongTensor([[1]]))

        output = output.permute(1, 0, 2).cpu().detach().numpy()

        output = np.argmax(output, axis=2)

        output = output[0]

        zh_data = []

        for w in output:
            w = str(w)
            if w in zh_idx2word.keys():
                zh_data.append(zh_idx2word[w])
                if zh_idx2word[w] == '<eos>':
                    break
            else:
                zh_data.append('<unk>')

        print(zh_data)