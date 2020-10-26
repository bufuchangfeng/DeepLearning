import spacy
from tqdm import tqdm
import os
import sys
from torch.utils.data import DataLoader
import copy
import numpy as np
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from seq2seq.model import *
from seq2seq.dataset import NMTDataset

spacy_en = spacy.load('en')
en_word2idx = None
zh_word2idx = None
device = None


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def preprocess(data):
    data = remove_space(data)
    data = format_str(data)
    data = list(data)
    return data


def remove_space(data):
    r = data.replace(' ', '')
    return r


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str


def padding_batch(batch):
    global device

    src_lens = [d["src_len"] for d in batch]
    trg_lens = [d["trg_len"] for d in batch]

    src_max = max([d["src_len"] for d in batch])
    trg_max = max([d["trg_len"] for d in batch])

    srcs = []
    trgs = []

    for d in batch:
        src = copy.deepcopy(d['src'])
        trg = copy.deepcopy(d['trg'])

        src.extend([en_word2idx["<pad>"]]*(src_max-d["src_len"]))
        trg.extend([zh_word2idx["<pad>"]]*(trg_max-d["trg_len"]))

        srcs.append(src)
        trgs.append(trg)

    srcs = torch.tensor(srcs, dtype=torch.long, device=device)
    trgs = torch.tensor(trgs, dtype=torch.long, device=device)

    batch = {"src":srcs.T, "src_lens":src_lens,
             "trg":trgs.T, "trg_lens":trg_lens}
    return batch


def main():
    with open('../data/cmn.txt', 'r', encoding='utf-8') as f:
        data = f.read()

    data = data.strip().split('\n')

    print('number of examples: ', len(data))

    en_data = [line.split('\t')[0] for line in data]
    zh_data = [line.split('\t')[1] for line in data]

    assert len(en_data) == len(zh_data)

    zh_words = set()
    en_words = set()

    for i in tqdm(range(len(zh_data))):
        en_seg = tokenize_en(en_data[i])
        zh_seg = preprocess(zh_data[i])

        zh_words.update(zh_seg)
        en_words.update(en_seg)

    global zh_word2idx
    zh_word2idx = {value: index + 4 for index, value in enumerate(zh_words)}

    zh_word2idx['<pad>'] = 0
    zh_word2idx['<sos>'] = 1
    zh_word2idx['<eos>'] = 2
    zh_word2idx['<unk>'] = 3

    global en_word2idx
    en_word2idx = {value: index + 4 for index, value in enumerate(en_words)}

    en_word2idx['<pad>'] = 0
    en_word2idx['<sos>'] = 1
    en_word2idx['<eos>'] = 2
    en_word2idx['<unk>'] = 3

    zh_idx2word = {zh_word2idx[k]: k for k in zh_word2idx.keys()}
    en_idx2word = {en_word2idx[k]: k for k in en_word2idx.keys()}

    with open('zh_word2idx', 'w+', encoding='utf-8') as f:
        json.dump(zh_word2idx, f)

    with open('en_word2idx', 'w+') as f:
        json.dump(en_word2idx, f)

    with open('zh_idx2word', 'w+', encoding='utf-8') as f:
        json.dump(zh_idx2word, f)

    with open('en_idx2word', 'w+') as f:
        json.dump(en_idx2word, f)

    zh = []
    en = []

    for i in tqdm(range(len(zh_data))):
        en_seg = tokenize_en(en_data[i])
        zh_seg = preprocess(zh_data[i])

        en_sentence = [en_word2idx['<sos>']] + [en_word2idx[w] for w in en_seg] + [en_word2idx['<eos>']]
        zh_sentence = [zh_word2idx['<sos>']] + [zh_word2idx[w] for w in zh_seg] + [zh_word2idx['<eos>']]

        en_len = len(en_sentence)
        zh_len = len(zh_sentence)

        zh.append({
            'sentence': zh_sentence,
            'len': zh_len
        })
        en.append({
            'sentence': en_sentence,
            'len': en_len
        })

    EPOCH = 700
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(len(en_word2idx), emb_dim=256, hid_dim=512, n_layers=4, dropout=0.5, bidirectional=True)
    decoder = Decoder(len(zh_word2idx), emb_dim=256, hid_dim=512, n_layers=4, dropout=0.5, bidirectional=True)
    model = Seq2Seq(encoder, decoder, device, max_trg_len=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    smooth = SmoothingFunction()

    # shuffle
    c = list(zip(en, zh))
    random.shuffle(c)
    en[:], zh[:] = zip(*c)

    train_dataset = NMTDataset(en, zh)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=padding_batch, shuffle=True)

    if os.path.exists('log'):
        os.remove('log')
    with open('log', 'w+') as f:
        pass

    best_valid_loss = sys.maxsize
    for epoch in range(EPOCH):

        print_bleu_total = 0
        print_loss_total = 0
        print_every = 10
        model.train()
        for index, batch in enumerate(train_loader):
            src = batch['src']
            src_lens = batch['src_lens']
            trg = batch['trg']
            trg_lens = batch['trg_lens']

            optimizer.zero_grad()
            output = model(src, src_lens, trg, trg_lens)

            # calculate bleu
            # bleu output [batch_size, seq_len, len(zh_word2idx)]
            bleu_output = copy.deepcopy(output.permute(1, 0, 2).cpu().detach().numpy())
            bleu_trg = copy.deepcopy(trg.permute(1, 0).cpu().detach().numpy())

            bleu_output = np.argmax(bleu_output, axis=2)
            for i in range(len(bleu_output)):
                candidate = copy.deepcopy(bleu_output[i])
                reference = copy.deepcopy(bleu_trg[i])

                candidate = [zh_idx2word[k] for k in candidate]
                reference = [[zh_idx2word[k] for k in reference]]

                _candidate = []
                for c in candidate:
                    _candidate.append(c)
                    if c == '<eos>':
                        break

                print_bleu_total += sentence_bleu(reference, _candidate, smoothing_function=smooth.method1)

            output = output.reshape(-1, len(zh_word2idx))
            trg = trg.reshape(-1).cpu()

            loss = criterion(output, trg)

            print_loss_total += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if index % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_bleu_avg = print_bleu_total / print_every / BATCH_SIZE
                print_loss_total = 0
                print_bleu_total = 0

                info = 'Train Epoch [{}/{}], Avg Loss: {:.4f} Bleu: {}'. \
                    format(epoch + 1, EPOCH, print_loss_avg, 100.0*print_bleu_avg)
                print(info)

                with open('log', 'a') as f:
                    f.write(info)
                    f.write('\n')

        valid_bleu = 0
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(train_loader):
                src = batch['src']
                src_lens = batch['src_lens']
                trg = batch['trg']
                trg_lens = batch['trg_lens']

                output = model(src, src_lens, trg, trg_lens, teacher_forcing_ratio=0)

                # calculate bleu
                # bleu output [batch_size, seq_len, len(zh_word2idx)]
                bleu_output = copy.deepcopy(output.permute(1, 0, 2).cpu().detach().numpy())
                bleu_trg = copy.deepcopy(trg.permute(1, 0).cpu().detach().numpy())

                bleu_output = np.argmax(bleu_output, axis=2)
                for i in range(len(bleu_output)):
                    candidate = copy.deepcopy(bleu_output[i])
                    reference = copy.deepcopy(bleu_trg[i])

                    candidate = [zh_idx2word[k] for k in candidate]
                    reference = [[zh_idx2word[k] for k in reference]]

                    _candidate = []
                    for c in candidate:
                        _candidate.append(c)
                        if c == '<eos>':
                            break

                    valid_bleu += sentence_bleu(reference, _candidate, smoothing_function=smooth.method1)

                output = output.reshape(-1, len(zh_word2idx))
                trg = trg.reshape(-1).cpu()

                loss = criterion(output, trg)

                valid_loss += loss.item()

            info = 'Train Epoch [{}/{}], Valid Loss: {:.4f} Bleu: {}'. \
                format(epoch + 1, EPOCH, valid_loss / len(train_loader), 100.0*valid_bleu / len(train_loader.dataset))
            print(info)

            with open('log', 'a') as f:
                f.write(info)
                f.write('\n')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()