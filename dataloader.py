import io
import logging
import threading
import queue
import traceback
import zipfile
from collections import defaultdict
import time
import os
import numpy as np
import torch
import pickle
import json
import glob
from utils.text import text_to_byte_sequence, text_to_char_sequence, text_to_word_sequence, text_to_phone_sequence, \
    sos_id, eos_id, get_aligned_words

np.random.seed(0)
zip_cache = {}

class PseudoZipfile:
    def __init__(self, dir):
        assert os.path.isdir(dir), "%s is not a directory" % dir
        dir = dir.strip(os.path.sep)
        self.dir = dir
        self.filename_set = set([l[len(dir) + 1:] for l in glob.glob(os.path.join(dir, '*'))])

    def open(self, filename, mode):
        filename = filename.replace('/', os.path.sep)
        if mode[-1] != 'b':
            mode += 'b'
        return open(os.path.join(self.dir, filename), mode)

def load_zip(filename):
    if os.path.isdir(filename):
        return PseudoZipfile(filename)
    if filename not in zip_cache:
        zip_cache[filename] = zipfile.ZipFile(filename)
        zip_cache[filename].filename_set = set(x.filename for x in zip_cache[filename].filelist)
    return zip_cache[filename]


class Feeder(threading.Thread):
    def __init__(self, zip_filename, metadata_file_path, hparams, spk_to_id=None, lang_to_id=None,
                 rank=0, world_size=1, adapt_lang=None, adapt_spk=None, train_lang=None, train_spk=None,
                 exclude_spk=None, downsample_lang=None, adapt_samples=None, warmup_lang=None, warmup_spk=None,
                 vocab=None, embed=None):
        super(Feeder, self).__init__()
        self._offset = 0
        self._epoch = 0
        self._spk_to_id = spk_to_id
        self._lang_to_id = lang_to_id
        self._hparams = hparams
        self.global_step = 1
        self.proto = get_input_proto(hparams)
        self.queue = queue.Queue(maxsize=2)
        self.rand = np.random.RandomState(rank)
        self._rank = rank
        self._world_size = world_size
        self._lock = threading.Lock()

        self.zfile = load_zip(zip_filename)
        logging.info("Found %d spectrograms" % len(self.zfile.filename_set))
        self.embed = None
        if embed:
            if hparams.use_external_embed:
                self.embed = load_zip(embed)
                logging.info("Found %d external embeddings" % len(self.embed.filename_set))
            else: # lexicon texts
                self.embed = json.load(open(embed, encoding='utf-8'))
        self.vocab = vocab

        # Load metadata
        with open(metadata_file_path, encoding='utf-8') as f:
            self._metadata = _read_meta(f, self._hparams.data_format, inc_lang=train_lang, inc_spk=train_spk)
        logging.info('%d samples read' % (len(self._metadata)))
        if exclude_spk:
            self._metadata = [m for m in self._metadata if m['n'].split('_')[0] not in exclude_spk]
            logging.info('%d samples after speakers excluded' % (len(self._metadata)))
        if downsample_lang:
            self._metadata = downsample_language(self._metadata, downsample_lang)
            logging.info('%d samples after language downsampling' % (len(self._metadata)))
        self._warmup_lang = warmup_lang
        self._warmup_spk = warmup_spk
        self._adapt_samples = adapt_samples

        hours = sum([int(x['l']) for x in self._metadata]) * hparams.frame_shift_ms / (3600 * 1000)
        logging.info('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))

        if self._world_size > 1:
            self._metadata = self._metadata[self._rank::self._world_size]
            logging.info("%d samples after sharding" % len(self._metadata))

        if self._hparams.shuffle_training_data:
            self.rand.shuffle(self._metadata)

        if hparams.balanced_training:
            logging.info('Using balanced data in training')
            self.grouped_meta = _group_meta(self._metadata, self._hparams)

        self._adapt_lang = adapt_lang
        self._adapt_spk = adapt_spk
        if self._adapt_lang or self._adapt_spk:
            with open(metadata_file_path, encoding='utf-8') as f:
                self._adapt_metadata = _read_meta(f, self._hparams.data_format,
                                                  inc_lang=adapt_lang, inc_spk=adapt_spk)
            logging.info('%d adapt samples read' % (len(self._adapt_metadata)))
            if exclude_spk:
                self._adapt_metadata = [m for m in self._adapt_metadata if m['n'].split('_')[0] not in exclude_spk]
                logging.info('%d adapt samples after speakers excluded' % (len(self._adapt_metadata)))
            if adapt_samples:
                self._adapt_metadata = [m for m in self._adapt_metadata if m['n'] in adapt_samples]
            elif downsample_lang:
                self._adapt_metadata = downsample_language(self._adapt_metadata, downsample_lang)
                logging.info('%d adapt samples after language downsampling' % (len(self._adapt_metadata)))
            spk_cnt = defaultdict(int)
            spk_time = defaultdict(int)
            for m in self._adapt_metadata:
                spk = m['n'].split('_')[0]
                spk_cnt[spk] += 1
                spk_time[spk] += int(m['l']) * hparams.frame_shift_ms / (60 * 1000)
            logging.info('Adapt samples by speakers: ' + ' '.join(
                ['%s (%d, %.3f min)' % (k, v, spk_time[k]) for k, v in spk_cnt.items()]))
            if self._world_size > 1:
                self._adapt_metadata = self._adapt_metadata[self._rank::self._world_size]
                logging.info('%d samples after language sharding' % (len(self._adapt_metadata)))
            if len(self._adapt_metadata) <= 30:
                logging.info('\n\t'.join(['Samples:'] + [m['n'] for m in self._adapt_metadata]))
            self._adapt_offset = 0
            self.rand.shuffle(self._adapt_metadata)
        else:
            self._adapt_metadata = None

    def run(self):
        try:
            while True:
                self._enqueue_next_group()
        except Exception:
            logging.error(traceback.format_exc())

    def state_dict(self):
        with self._lock:
            state = {'rand': self.rand.get_state()}
            if self._hparams.balanced_training:
                state['offset'] = self.grouped_meta['offsets']
                state['epoch'] = self.grouped_meta['epoch']
            else:
                state['offset'] = self._offset
                state['epoch'] = self._epoch

            if hasattr(self, '_adapt_offset'):
                state['adapt_offset'] = self._adapt_offset
            logging.info("Dumped feeder state: " + str(state['offset']))
            return state

    def load_state_dict(self, state):
        logging.info("Loaded feeder state: " + str(state['offset']))
        self.rand.set_state(state['rand'])
        if self._hparams.balanced_training:
            self.grouped_meta['offsets'] = state['offset']
            self.grouped_meta['epoch'] = state['epoch']
        else:
            self._offset = state['offset']
            self._epoch = state['epoch']
        if hasattr(self, '_adapt_offset'):
            state['adapt_offset'] = self._adapt_offset


    def get_examples(self, bucket_size):
        examples = []
        with self._lock:
            for i in range(bucket_size):
                examples.append(self._get_next_example())
        return examples

    def get_batch(self):
        return self.queue.get()

    def _cast_tensor(self, batch):
        batch = dict([(name, batch[name]) for name in self.proto])
        if self._world_size > 1 and torch.cuda.is_available():  # Reduce memory cost; support DDP only
            ctx = torch.cuda.device(self._rank)
        else:
            ctx = memoryview(b'')  # no-op
        with ctx:
            for key in batch:
                batch[key] = self.proto[key](batch[key])
                if isinstance(batch[key], torch.Tensor) and torch.cuda.is_available():
                    batch[key] = batch[key].pin_memory()
        return batch

    def _enqueue_next_group(self):
        tic = time.time()
        examples = self.get_examples(self._hparams.bucket_size)
        examples.sort(key=lambda x: len(x['mel_target']))
        batches = _pack_into_batches(examples, hparams=self._hparams)
        self.rand.shuffle(batches)

        for batch in batches:
            batch = _prepare_batch(batch, hparams=self._hparams, ex_embed=self.embed, vocab=self.vocab)
            batch = self._cast_tensor(batch)
            self.queue.put(batch)
        logging.info("Packed %d batches with %d samples in %.2f sec" % (len(batches), len(examples), time.time() - tic))

    def _get_next_balanced_meta(self):
        lang = self.rand.choice(self.grouped_meta['langs'], p=self.grouped_meta['prob'])
        meta = self.grouped_meta['meta'][lang][self.grouped_meta['offsets'][lang]]
        self.grouped_meta['offsets'][lang] += 1
        if self.grouped_meta['offsets'][lang] >= len(self.grouped_meta['meta'][lang]):
            self.grouped_meta['offsets'][lang] = 0
            self.grouped_meta['epoch'][lang] += 1
            logging.info("Start epoch %d of %s" % (self.grouped_meta['epoch'][lang], lang))
        return meta

    def _get_next_example(self):
        while True:
            if self._adapt_metadata and self.rand.random() < self._adapt_rate():
                meta = self._adapt_metadata[self._adapt_offset]
                self._adapt_offset += 1
                if self._adapt_offset >= len(self._adapt_metadata):
                    self._adapt_offset = 0
                    self.rand.shuffle(self._adapt_metadata)
            elif not self._hparams.balanced_training:
                meta = self._metadata[self._offset]
                self._offset += 1
                if self._offset >= len(self._metadata):
                    self._offset = 0
                    self._epoch += 1
                    if self._hparams.shuffle_training_data:
                        self.rand.shuffle(self._metadata)
            else:
                meta = self._get_next_balanced_meta()

            if self.skip_meta(meta):
                continue
            break

        return extract_meta(meta, self.zfile, self._hparams, self._spk_to_id, self._lang_to_id, vocab=self.vocab)

    def _adapt_rate(self):
        if self.global_step >= self._hparams.adapt_end_step:
            r = 1.0
        elif self.global_step < self._hparams.adapt_start_step:
            r = 0.0
        else:
            r = (self.global_step - self._hparams.adapt_start_step) / \
                (self._hparams.adapt_end_step - self._hparams.adapt_start_step)
        return r * self._hparams.final_adapt_rate

    def skip_meta(self, meta):
        if self.global_step >= self._hparams.data_warmup_steps:
            return False
        if self._warmup_lang is not None and meta.get('i', None) not in self._warmup_lang:
            return True
        if self._warmup_spk is not None and meta['n'].split('_')[0] not in self._warmup_spk:
            return True
        if self._hparams.target_length_upper_bound < 0 or \
                self._hparams.target_length_lower_bound <= int(meta['l']) <= self._hparams.target_length_upper_bound:
            return False
        return True


class FeederEval:
    def __init__(self, zip_filename, metadata_file_path, hparams, spk_to_id=None, lang_to_id=None,
                 eval_lang=None, eval_spk=None, exclude_spk=None, target_lang=None, target_spk=None,
                 shuffle=True, keep_order=False, pick_partial=False, single=False, vocab=None, embed=None):
        super(FeederEval, self).__init__()
        self._offset = 0
        self._shuffle = shuffle
        self._keep_order = keep_order
        self.single = single
        self.lang_ids = lang_to_id
        self.spk_ids = spk_to_id
        self._target_lang = target_lang
        self._target_spk = target_spk
        self._eval_lang = eval_lang
        self._eval_spk = eval_spk
        self._hparams = hparams
        self.proto = get_input_proto(hparams)

        self.zfile = load_zip(zip_filename) if zip_filename is not None else None
        self.embed = None
        if embed:
            if hparams.use_external_embed:
                self.embed = load_zip(embed)
                logging.info("Found %d external embeddings" % len(self.embed.filename_set))
            else: # lexicon texts
                self.embed = json.load(open(embed, encoding='utf-8'))
        self.vocab = vocab

        with open(metadata_file_path, encoding='utf-8') as f:
            self._metadata = _read_meta(f, self._hparams.data_format, inc_lang=eval_lang, inc_spk=eval_spk)
        logging.info('%d eval samples read' % len(self._metadata))

        if 'l' in hparams.data_format:
            self._metadata = [m for m in self._metadata if int(m['l']) < hparams.max_eval_sample_length]
            logging.info('%d eval samples after filtering length' % len(self._metadata))

        if exclude_spk:
            self._metadata = [m for m in self._metadata if m['n'].split('_')[0] not in exclude_spk]
            logging.info('%d eval samples after speakers excluded' % (len(self._metadata)))
        if pick_partial:
            self._metadata = filter_eval_samples(self._metadata, 3, self._hparams.eval_sample_per_speaker)
            logging.info('%d eval samples after filtering' % len(self._metadata))
        self._meta_texts = ['|'.join([m[c] for c in self._hparams.data_format]) for m in self._metadata]

        self.data = self.prepare_all_batches(self.get_all_batches())
        self.rand = np.random.RandomState(0)
        if self._shuffle:
            self.rand.shuffle(self.data)
        logging.info('[FeederEval] Prepared %d batches' % len(self.data))

    def fetch_data(self, exclude=None):
        if exclude is None:
            data = self.data
        else:
            data = self.prepare_all_batches(self.get_all_batches(exclude))
        if self._shuffle and not self._keep_order:
            self.rand.shuffle(data)
        for batch in data:
            for name in batch:
                if name in self.proto:
                    batch[name] = self.proto[name](batch[name])
        return data

    def _get_next_example(self):
        finished = False
        meta = self._metadata[self._offset]
        self._offset += 1
        if self._offset >= len(self._metadata):
            self._offset = 0
            finished = True

        return extract_meta(meta, self.zfile, self._hparams, self.spk_ids, self.lang_ids,
                            target_spk=self._target_spk, target_lang=self._target_lang, vocab=self.vocab), finished

    def _get_all_examples(self):
        examples = []
        while True:
            example, finished = self._get_next_example()
            examples.append(example)
            if finished:
                break
        return examples

    def get_all_batches(self, exclude=[]):
        examples = self._get_all_examples()
        examples = [x for x in examples if x['name'] not in exclude]

        if 'mel_target' in examples[0]:
            examples.sort(key=lambda x: len(x['mel_target']))
        batches = _pack_into_batches(examples, self.single, hparams=self._hparams)
        return batches

    def prepare_all_batches(self, batches):
        ret = []
        for batch in batches:
            batch = _prepare_batch(batch, hparams=self._hparams, ex_embed=self.embed, vocab=self.vocab)
            ret.append(batch)
        return ret


def _read_meta(meta_file, format, inc_lang=None, inc_spk=None):
    meta_list = []
    for line in meta_file:
        parts = line.strip().split('|')
        if len(parts) != len(format):
            parts = line.strip().split('\t')
        if format == 'nlti':
            name, length, text, lang = parts
            item_dict = {'n': name, 'l': length, 't': text, 'i': lang}
        elif format == 'nltpi':
            name, length, text, phone, lang = parts
            item_dict = {'n': name, 'l': length, 't': text, 'p': phone, 'i': lang}
        else:
            raise ValueError('Invalid format for _read_meta: %s' % format)
        if inc_lang is not None and lang not in inc_lang:
            continue
        if inc_spk is not None and name.split('_')[0] not in inc_spk:
            continue
        meta_list.append(item_dict)
    return meta_list


def _group_meta(metadata, hparams):
    lang_meta = defaultdict(list)
    lang_spk = defaultdict(set)
    for m in metadata:
        lang_meta[m['i']].append(m)
        lang_spk[m['i']].add(m['n'].split('_')[0])
    langs = list(lang_meta.keys())
    langs.sort()
    sizes = [len(lang_meta[l]) for l in langs]
    alphas = np.power(np.asarray(sizes) / np.sum(sizes), hparams.lg_prob_scale)
    prob = alphas / np.sum(alphas)
    for i, lang in enumerate(langs):
        logging.info("\t%s: %d samples, prob=%f" % (lang, sizes[i], prob[i]))
        spks = list(lang_spk[lang])
        spks.sort()
        logging.info('\tSpeakers: ' + str(spks))
    return {'langs': langs, 'prob': prob, 'meta': lang_meta,
            'offsets': dict([(l, 0) for l in langs]), 'epoch': dict([(l, 0) for l in langs])}


def downsample_language(meta_list, downsample_langs):
    mark = [True for _ in meta_list]
    lang_bins = defaultdict(list)
    for i, m in enumerate(meta_list):
        if m['i'] in downsample_langs:
            lang_bins[m['i']].append(i)
    for lang_key, values in lang_bins.items():
        r = np.random.RandomState(0)
        r.shuffle(values)
        if downsample_langs[lang_key] <= 1:
            keep_samples = int(len(values) * downsample_langs[lang_key])
        else:
            keep_samples = int(downsample_langs[lang_key])
        for i in range(keep_samples, len(values)):
            mark[values[i]] = False

    meta_list = [meta_list[k] for k in range(len(mark)) if mark[k]]
    return meta_list


def filter_eval_samples(meta, n_spk, n_sample, required_speakers=None):
    lang_samples = defaultdict(list)
    for m in meta:
        lang_samples[m['i']].append(m)
    samples = []
    for lang in lang_samples:
        r = np.random.RandomState(0)
        r.shuffle(lang_samples[lang])
        spk_cnt = {}
        if required_speakers is not None:
            n_spk = len(required_speakers)
            for s in required_speakers:
                spk_cnt[s] = 0
        for m in lang_samples[lang]:
            spk = m['n'].split('_')[0]
            if spk not in spk_cnt:
                if len(spk_cnt) >= n_spk:
                    continue
                spk_cnt[spk] = 0
            spk_cnt[spk] += 1
            if spk_cnt[spk] <= n_sample:
                samples.append(m)
    r = np.random.RandomState(0)
    r.shuffle(samples)
    return samples


def _pack_into_batches(examples, single=False, hparams=None):
    batches = [[]]
    batch_max_input_len = 0
    for sample in examples:
        target_len = len(sample['mel_target']) if 'mel_target' in sample else int(len(sample['input']) * 10)
        batch_max_input_len = max(batch_max_input_len, len(sample['input']))
        quad_cnt = batch_max_input_len ** 2 + target_len ** 2
        if (len(batches[-1]) + 1) * quad_cnt > hparams.batch_frame_quad_limit or \
                (len(batches[-1]) + 1) * target_len > hparams.batch_frame_limit or single \
                or len(batches[-1]) == hparams.max_batch_size:
            batches.append([])
            batch_max_input_len = len(sample['input'])
        batches[-1].append(sample)
    return batches


def _load_from_zip(zfile, npy_name):
    with zfile.open(npy_name, 'r') as zip_npy:
        with io.BytesIO(zip_npy.read()) as raw_npy:
            return np.load(raw_npy)


embed_cache = {}
embed_cache_list = []
def get_embed(key, embed_file, lang):
    key = str(key) + '.pickle'
    if lang == 'ja-jp':
        lang = 'ja'
    if lang in ['ja', 'zh-hk'] and lang is not None:
        key = lang + '/' + key
    if key not in embed_file.filename_set:
        return {'tokens': [''], 'key': np.zeros([1, 1024], dtype=np.float32),
                'value': np.zeros([1, 2048], dtype=np.float32), 'empty': True}
    if key in embed_cache:
        return embed_cache[key]
    else:
        with embed_file.open(key, 'r') as pf:
            embed = pickle.loads(pf.read())
        embed = {'tokens': embed['tokens'], 'key': np.squeeze(embed['input'], axis=0),
                 'value': np.squeeze(np.concatenate([embed['input'], embed['output']], axis=-1), axis=0)}
        embed_cache[key] = embed
        embed_cache_list.append(key)
        if len(embed_cache_list) > 15000:
            del embed_cache[embed_cache_list[0]]
            del embed_cache_list[0]
    return embed

def get_char_embed(vocab, lexicon, key, context_size):
    if key in lexicon['gloss']:
        text = lexicon['gloss'][key][:200].replace('●', '*')
        if text.startswith("释义："):
            text = text[3:]
        toks = [[vocab.get(ch, 0)] for ch in text]
    else:
        text = ['']
        toks = [[0]]
    return {'tokens': list(text), 'key': np.asarray(toks, dtype=np.int32),
            'value': np.zeros([0, context_size], dtype=np.float32), 'empty': key not in lexicon['gloss']}

def get_lexicon_embed(all_words, all_langs, embeddings, vocab, key_size, value_size, use_external_embed):
    # words: List of list of word of each token
    keys = [np.zeros([1, key_size], dtype=np.float32 if use_external_embed else np.int32)]  # [n_entry, length_k, depth_k]
    contexts = [np.zeros([1, value_size], dtype=np.float32)]  # [n_entry, length_k, depth_v]
    context_tokens = [['']]
    batch_scripts = []  # [batch, length_q, length_k]
    context_lengths = [1]  # [n_entry]
    word_index = {None: 0} # Placeholder
    indices = []
    for wi, words in enumerate(all_words):
        indices.append([])
        batch_scripts.append([])
        for w in words:
            key = (w, all_langs[wi])
            if key not in word_index:
                if use_external_embed:
                    wid = vocab[w]
                    embed = get_embed(str(wid), embeddings, all_langs[wi])
                else:
                    embed = get_char_embed(vocab, embeddings, w, context_size=value_size)
                if embed.get("empty", False):
                    idx = 0
                else:
                    idx = word_index[key] = len(word_index)
                    keys.append(embed['key'])
                    contexts.append(embed['value'])
                    context_lengths.append(len(embed['key']))
                    context_tokens.append(embed['tokens'])
            else:
                idx = word_index[key]
            indices[-1].append(idx)
            batch_scripts[-1].append(context_tokens[idx])
    return keys, contexts, context_lengths, indices, batch_scripts


def _prepare_batch(batch, hparams, ex_embed=None, vocab=None):
    inputs = _prepare_inputs([x['input'] for x in batch])
    input_lengths = np.asarray([len(x['input']) for x in batch], dtype=np.int32)
    results = {'inputs': inputs, 'input_lengths': input_lengths}

    if 'target_length' in batch[0]:
        target_lengths = np.asarray([x['target_length'] for x in batch], dtype=np.int32)
        results['target_lengths'] = target_lengths
    elif 'mel_target' in batch[0]:
        target_lengths = np.asarray([len(x['mel_target']) for x in batch], dtype=np.int32)
        results['target_lengths'] = target_lengths
    if 'mel_target' in batch[0]:
        mel_targets = _prepare_targets([x['mel_target'] for x in batch])
        results['mel_targets'] = mel_targets


    if hparams.multi_lingual:
        results['input_language_vecs'] = np.asarray([x['language_vec'] for x in batch], dtype=np.float32)
    if hparams.multi_speaker or hparams.multi_lingual:
        results['input_spk_ids'] = np.asarray([x['speaker_id'] for x in batch], dtype=np.float32)
    if hparams.use_knowledge_attention:
        keys, contexts, context_lengths, context_indices, context_script = \
            get_lexicon_embed([x['words'] for x in batch], [x['lang'] for x in batch], ex_embed, vocab,
                              hparams.knowledge_key_size if hparams.use_external_embed else 1,
                              hparams.knowledge_value_size, hparams.use_external_embed)
        if hparams.use_external_embed:
            results['keys'] = _prepare_targets(keys)
        else:
            results['keys'] = _prepare_targets(keys)
        results['contexts'] = _prepare_targets(contexts)
        results['context_indices'] = _prepare_inputs([np.asarray(x) for x in context_indices])
        results['context_lengths'] = np.asarray(context_lengths)
        results['context_scripts'] = context_script

    results['input_scripts'] = [x['input_scripts'] for x in batch]
    results['langs'] = [x['lang'] for x in batch]
    results['names'] = [x['name'] for x in batch]
    return results


def _prepare_inputs(inputs):
    max_len = max([len(x) for x in inputs])
    return np.stack([_pad_input(x, max_len) for x in inputs])

import numba as nb
from numba import prange
from numba.typed import List

def _prepare_targets_nojit(targets):
    max_len = max([len(t) for t in targets])
    result = np.zeros([len(targets), max_len, targets[0].shape[-1]])
    for i in range(len(targets)):
        result[i, :targets[i].shape[0]] = targets[i]
    return result

@nb.jit(nopython=True, parallel=True)
def _prepare_targets_jit(targets):
    max_len = max([int(t.shape[0]) for t in targets])
    result = np.empty((len(targets), max_len, targets[0].shape[-1]), dtype=targets[0].dtype)
    for i in prange(len(targets)):
        result[i, :targets[i].shape[0]] = targets[i]
        result[i, targets[i].shape[0]:] = 0
    return result

def _prepare_targets(targets):
    typed_a = List()
    [typed_a.append(x) for x in targets]
    return _prepare_targets_jit(typed_a)

@nb.jit(nopython=True, parallel=True)
def _prepare_targets_jit_int(targets):
    max_len = max([int(t.shape[0]) for t in targets])
    result = np.empty((len(targets), max_len, targets[0].shape[-1]), dtype=targets[0].dtype)
    for i in prange(len(targets)):
        result[i, :targets[i].shape[0]] = targets[i]
        result[i, targets[i].shape[0]:] = 0
    return result

def _prepare_targets_int(targets):
    typed_a = List()
    [typed_a.append(x) for x in targets]
    return _prepare_targets_jit(typed_a)


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=0)


def extract_meta(meta, zfile, hparams, spk_ids, lang_ids, target_spk=None, target_lang=None, vocab=None):
    name = meta['n']
    if name.endswith('.npy'):
        name = name[:-4]
    results = {'name': name}
    if zfile:
        mel_target = _load_from_zip(zfile, meta['n'])
    else:
        mel_target = None
    if mel_target is not None:
        if 'l' in meta:
            target_length = int(meta['l'])
        else:
            target_length = mel_target.shape[0]
        results['mel_target'] = mel_target
        results['target_length'] = target_length

    if target_lang is not None:
        lang = target_lang
    else:
        lang = meta.get('i', None)
    results['lang'] = lang
    if hparams.multi_lingual and lang:
        language_vec = np.zeros([hparams.max_num_language])
        language_vec[lang_ids[lang]] = 1
        results['language_vec'] = language_vec

    results['input_scripts'] = meta['t']
    if hparams.input_method == 'byte':
        input_data, offsets = text_to_byte_sequence(meta['t'])
    elif hparams.input_method == 'phone':
        input_data = text_to_phone_sequence(meta['p'], vocab)
        results['input_scripts'] = meta['p']
        offsets = []
    elif hparams.input_method == 'char':
        input_data, offsets = text_to_char_sequence(meta['t'], vocab, remove_space=hparams.remove_space)
    elif hparams.input_method == 'word':
        input_data, offsets = text_to_word_sequence(meta['t'], vocab)
    input_data = [sos_id] + input_data + [eos_id]
    offsets = [-1] + offsets + [-1]

    results['input'] = np.asarray(input_data, dtype=np.int32)
    if hparams.use_knowledge_attention:
        token_to_word = get_aligned_words(meta['t'], vocab, offsets)
        results['words'] = token_to_word

    if hparams.multi_speaker or hparams.multi_lingual:
        if target_spk:
            speaker_id = spk_ids[target_spk]
        else:
            speaker_id = spk_ids[name.split('_')[0]]
        results['speaker_id'] = speaker_id
    return results


def get_input_proto(config):
    keys = {'inputs': torch.LongTensor, 'input_lengths': torch.LongTensor,
            'mel_targets': torch.FloatTensor, 'target_lengths': torch.LongTensor,
            'names': list, 'input_scripts': list, 'langs': list}
    if config.multi_speaker or config.multi_lingual:
        keys['input_spk_ids'] = torch.LongTensor
    if config.multi_lingual:
        keys['input_language_vecs'] = torch.FloatTensor
    if config.use_knowledge_attention:
        if config.use_external_embed:
            keys['keys'] = torch.FloatTensor
        else:
            keys['keys'] = torch.LongTensor
        keys['contexts'] = torch.FloatTensor
        keys['context_lengths'] = torch.LongTensor
        keys['context_indices'] = torch.LongTensor
        keys['context_scripts'] = list
    return keys
