import requests
import json
from utils.text import is_sep
import os
from lexicon.lexicon_utils import base_path, cache_path, get_page, get_text
import tqdm
from bs4 import BeautifulSoup
import unicodedata
from lexicon.encode import get_all_encoding
from lexicon.read_zdic import *

cache_path = os.path.join(cache_path, 'zdic')
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

def build_vocab(meta_path, base_path, out_path):
    meta = open(meta_path, encoding='utf-8').read().splitlines()
    meta = [m.split('|') for m in meta]
    meta = [m for m in meta if m[0].startswith('databaker')]

    if base_path is None:
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        n = 3
    else:
        vocab = json.load(open(base_path, encoding='utf-8'))
        n = max(vocab.values()) + 1

    for m in meta:
        text = m[2]
        for ch in text:
            if ch not in vocab:
                vocab[ch] = n
                n += 1

    json.dump(vocab, open(out_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

if __name__ == '__main__':
    build_vocab(r'D:\free_corpus\packed\metadata.txt', None,
                r'D:\free_corpus\vocab\db_vocab.json')
    fetch_gloss(r'D:\free_corpus\vocab\db_vocab.json', None, r'D:\free_corpus\vocab\db_lexicon.json')
    word_id = json.load(open(r"D:\free_corpus\vocab\db_vocab.json", encoding='utf-8'))
    lexicon = json.load(open(r'D:\free_corpus\vocab\db_lexicon.json', encoding='utf-8'))
    get_all_encoding(lexicon['gloss'].items(), r"D:\free_corpus\embed\db", word_id)