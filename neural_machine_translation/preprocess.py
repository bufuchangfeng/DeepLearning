import jieba
from tqdm import tqdm
import spacy
import json

spacy_en = spacy.load('en')


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def build_word_dict():
    zh_words = set()
    en_words = set()

    en_max_len = -1
    zh_max_len = -1

    with open('./data/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines[:100000]):
            line = line.split('\t')
            en, zh = line[2], line[3].replace('\n', '').strip()

            en_seg = tokenize_en(en)
            zh_seg = list(jieba.cut(zh))

            if len(en_seg) > en_max_len:
                en_max_len = len(en_seg)

            if len(zh_seg) > zh_max_len:
                zh_max_len = len(zh_seg)

            zh_words.update(zh_seg)
            en_words.update(en_seg)

    zh_word2idx = {value: index + 4 for index, value in enumerate(zh_words)}

    zh_word2idx['<pad>'] = 0
    zh_word2idx['<sos>'] = 1
    zh_word2idx['<eos>'] = 2
    zh_word2idx['<unk>'] = 3

    en_word2idx = {value: index + 4 for index, value in enumerate(en_words)}

    en_word2idx['<pad>'] = 0
    en_word2idx['<sos>'] = 1
    en_word2idx['<eos>'] = 2
    en_word2idx['<unk>'] = 3

    zh_idx2word = {zh_word2idx[k]: k for k in zh_word2idx.keys()}
    en_idx2word = {en_word2idx[k]: k for k in en_word2idx.keys()}

    metadata = {
        'zh_max_len': zh_max_len,
        'en_max_len': en_max_len
    }

    with open('./data/zh_word2idx', 'w+', encoding='utf-8') as f:
        json.dump(zh_word2idx, f)

    with open('./data/en_word2idx', 'w+') as f:
        json.dump(en_word2idx, f)

    with open('./data/zh_idx2word', 'w+', encoding='utf-8') as f:
        json.dump(zh_idx2word, f)

    with open('./data/en_idx2word', 'w+') as f:
        json.dump(en_idx2word, f)

    with open('./data/metatdata', 'w+') as f:
        json.dump(metadata, f)


def main():
    build_word_dict()


if __name__ == '__main__':
    main()
