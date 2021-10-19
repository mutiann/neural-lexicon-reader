import transformers
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
import torch
import json
import pickle
import tqdm
import os
from matplotlib import pyplot as plt
lengths = []

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
model = XLMRobertaModel.from_pretrained("xlm-roberta-large")
max_token = 256
def shrink_text(text, inputs):
    print(text)
    char_p = []
    for i in range(len(text)):
        if text[i] == '●' and i > 0:
            char_p.append(i)
    char_p.append(len(text) - 1)

    reduction = inputs.data['input_ids'].shape[1] - max_token + 1
    offset = inputs.data['offset_mapping'][0].numpy().tolist()[1:-1]

    tok_p = []
    seg_len = []
    for cp in char_p:
        for i, (l, r) in enumerate(offset):
            if l <= cp < r:
                if len(tok_p) == 0:
                    seg_len.append(i)
                else:
                    seg_len.append(i - tok_p[-1])
                tok_p.append(i)
                break
    remove_toks = [0 for _ in seg_len]
    while reduction > 0:
        maxp = maxv = 0
        for i in range(len(seg_len)):
            if seg_len[i] > maxv:
                maxv = seg_len[i]
                maxp = i
        seg_len[maxp] -= 1
        remove_toks[maxp] += 1
        reduction -= 1

    for i, n_toks in reversed(list(enumerate(remove_toks))):
        if n_toks > 0:
            l = offset[tok_p[i] - n_toks][1]
            r = offset[tok_p[i]][0]
            if tok_p[i] == len(offset) - 1:
                r = len(text)
            text = text[:l] + '…' + text[r:]
    print(text)
    return text

def get_encodings(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        if inputs.data['input_ids'].shape[1] >= max_token:
            text = shrink_text(text, inputs)
            inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        lengths.append(inputs.data['input_ids'].shape[1])
        del inputs['offset_mapping']
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        return {'input': model.get_input_embeddings()(inputs['input_ids']).cpu().numpy(),
                'output': outputs['hidden_states'][-1].cpu().numpy(),
                'tokens': ['<sos>'] + tokenizer.tokenize(text) + ['<eos>']}

def get_all_encoding(lexicon, out_path, word_id):
    os.makedirs(out_path, exist_ok=True)
    # skipped = {}
    lengths = []
    for key, text in tqdm.tqdm(lexicon):
        length = len(tokenizer.tokenize(text))
        lengths.append(length)
        key_id = word_id[key]
        if os.path.exists(os.path.join(out_path, str(key_id) + '.pickle')):
            continue
        text = text.replace("*", '●')
        encoding = get_encodings(text)
        pickle.dump(encoding, open(os.path.join(out_path, str(key_id) + '.pickle'), 'wb'))
    # print(json.dumps(skipped, indent=1, ensure_ascii=False))
    print("%d %d %d" % (len([t for t in lengths if t >= 200]), len([t for t in lengths if t >= 300]), len([t for t in lengths if t >= 400])))
    plt.hist(lengths)
    plt.show()
