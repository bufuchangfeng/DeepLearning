# 使用了 10 0000 条数据

import json
import _pickle as pickle
from tqdm import tqdm
import spacy
import jieba


spacy_en = spacy.load('en')


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def convert_data():

    f = open('./data/zh_word2idx', 'r+', encoding='utf-8')
    zh_word2idx = json.load(f)
    f.close()

    f = open('./data/en_word2idx', 'r+')
    en_word2idx = json.load(f)
    f.close()

    f = open('./data/metatdata', 'r+')
    metadata = json.load(f)
    f.close()

    zh_max_len = metadata['zh_max_len'] + 2
    en_max_len = metadata['en_max_len'] + 2

    zh_data = []
    en_data = []
    with open('./data/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines[:100000]):
            line = line.split('\t')
            en, zh = line[2], line[3].replace('\n', '').strip()

            en_seg = tokenize_en(en)
            zh_seg = list(jieba.cut(zh))

            en_sentence = [en_word2idx['<sos>']] + [en_word2idx[w] for w in en_seg] + [en_word2idx['<eos>']]
            zh_sentence = [zh_word2idx['<sos>']] + [zh_word2idx[w] for w in zh_seg] + [zh_word2idx['<eos>']]

            en_diff = en_max_len - len(en_sentence)
            zh_diff = zh_max_len - len(zh_sentence)

            if en_diff > 0:
                en_sentence += [en_word2idx['<pad>']] * en_diff
            if zh_diff > 0:
                zh_sentence += [zh_word2idx['<pad>']] * zh_diff

            zh_data.append(zh_sentence)
            en_data.append(en_sentence)

    f = open('./data/zh_data', 'wb+')
    pickle.dump(zh_data, f)
    f.close()

    f = open('./data/en_data', 'wb+')
    pickle.dump(en_data, f)
    f.close()


def main():
    convert_data()


if __name__ == '__main__':
    main()