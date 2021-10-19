import logging
import unicodedata
pad_id = 0
eos_id = 1
sos_id = 2

def is_sep(ch):
    if unicodedata.category(ch) in ["Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Zl", "Zp", "Zs"]:
        return True
    return False


def is_sep(ch):
    if unicodedata.category(ch) in ["Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Zl", "Zp", "Zs"]:
        return True
    return False

# offset: token -> char
def text_to_byte_sequence(text: str):
    s = []
    offsets = []
    for i, ch in enumerate(text):
        ch = ch.encode('utf-8')
        s += list(ch)
        offsets.extend([i] * len(ch))
    return s, offsets

def text_to_phone_sequence(phones: str, phone_set: dict):
    phones = phones.split(' ')
    return [phone_set[p] for p in phones]

def text_to_char_sequence(text: str, vocab: dict, remove_space):
    offsets = []
    tokens = []
    for i in range(len(text)):
        if remove_space and text[i] == ' ':
            continue
        offsets.append(i)
        tokens.append(vocab[text[i]])
    return tokens, offsets

def split_words(text):
    words = ['']
    offset = [0]
    for i, ch in enumerate(text):
        if is_sep(ch):
            if words[-1] == '':
                words[-1] = ch
            else:
                words.append(ch)
                offset.append(i)
            words.append('')
            offset.append(i+1)
        else:
            words[-1] = words[-1] + ch
    if words[-1] == '':
        words = words[:-1]
        offset = offset[:-1]
    return words, offset

def split_unk_words(words, offsets, vocab):
    tokens = []
    offsets_ = []
    for i, word in enumerate(words):
        if word in vocab or len(word) == 1:
            tokens.append(word)
            offsets_.append(offsets[i])
        else:
            tokens.extend(list(word))
            offsets_.extend([offsets[i] + k for k in range(len(word))])
    return tokens, offsets_

def text_to_word_sequence(text: str, vocab: dict):
    words, offset = split_words(text)
    tokens, offset = split_unk_words(words, offset, vocab)
    tokens = [vocab[t] if t in vocab else 0 for t in tokens]
    return tokens, offset

def get_aligned_words(text: str, vocab: dict, token_offset):
    words, offset = split_words(text)
    words, offset = split_unk_words(words, offset, vocab)

    ch_to_word = [None for _ in range(len(text))]
    for i in range(len(offset)):
        ch_to_word[offset[i]] = i
    last_wi = -1
    for i in range(len(text)):
        if ch_to_word[i] is not None:
            last_wi = ch_to_word[i]
        ch_to_word[i] = last_wi

    token_to_word = []
    for ch_i in token_offset[1: -1]:
        wi = ch_to_word[ch_i]
        token_to_word.append(words[wi])
    token_to_word = ['<sos>'] + token_to_word + ['<eos>']
    return token_to_word

id_to_token = None
def sequence_to_text(tokens, vocab):
    global id_to_token
    if id_to_token is None:
        id_to_token = dict([(v, k) for k, v in vocab.items()])
    return ''.join([id_to_token[id] for id in tokens])

def language_name_to_id(lang_to_id, lang):
    id_to_lang = dict([(v, k) for k, v in lang_to_id.items()])
    if isinstance(lang, str):
        lang = lang.split(':')
    else:
        lang = list(lang)
    for i in range(len(lang)):
        if lang[i].isnumeric():
            if lang[i] not in id_to_lang:
                logging.warn('Unknown language requested: ' + str(lang[i]))
        else:
            if lang[i] in lang_to_id:
                lang[i] = lang_to_id[lang[i]]
            else:
                logging.warn('Unknown language requested: ' + str(lang[i]))
    lang = [t for t in lang if t in id_to_lang]
    logging.info('Selected languages: ' + ' '.join([id_to_lang[t] for t in lang]))
    return lang

def language_vec_to_id(lv):
    for i in range(len(lv)):
        if lv[i] > 0:
            return i
    return -1