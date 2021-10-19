import os
import torch
import logging
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt
import matplotlib.font_manager as mfm
import pickle

def set_logger(output_path=None, name=None):
    fmt = logging.Formatter("[" + (name + ' ' if name else '') + "%(levelname)s %(asctime)s]" + " %(message)s")
    handlers = []
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    h.setLevel(logging.INFO)
    handlers.append(h)
    if output_path is not None:
        h = logging.FileHandler(output_path, 'a', 'utf-8')
        h.setFormatter(fmt)
        h.setLevel(logging.INFO)
        handlers.append(h)
    if len(logging.root.handlers) == 0:
        logging.basicConfig(handlers=handlers, level=logging.INFO)
        logging.info('logging set: ' + str(logging.root.handlers))
    else:
        logging.warn('logging is already used: ' + str(logging.root.handlers))
        while logging.root.hasHandlers():
            logging.root.removeHandler(logging.root.handlers[0])
        logging.root.setLevel(logging.INFO)
        for h in handlers:
            logging.root.addHandler(h)

def plot_mel(path, mel, title=''):
    if mel.shape[-1] == 80:
        mel = mel.T
    fig, ax = plt.subplots()
    ax.pcolormesh(mel)
    if title:
        ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)

def plot_attn(attn, path, enc_length=None, dec_length=None):
    # attn: [(heads, dec, enc)]
    results = None
    best_score = 0
    info = ''
    for k, layer_attn in enumerate(attn):
        if enc_length:
            layer_attn = layer_attn[:, :, :enc_length]
        if dec_length:
            layer_attn = layer_attn[:, :dec_length]
        for head in range(layer_attn.shape[0]):
            score = 0
            for dec_step in range(layer_attn.shape[1]):
                score += layer_attn[head, dec_step].max()
            if score > best_score:
                results = layer_attn[head]
                best_score = score
                info = "Layer %d, Head %d" % (k, head)
    fig, ax = plt.subplots()
    fig.set_size_inches(45, 15)
    ax.pcolormesh(results)
    ax.set_title(info)
    fig.savefig(path)
    plt.close(fig)


def plot_single_attn_with_label(x_script, y_script, path, attn, k, head):
    info = "Layer %d, Head %d" % (k, head)
    if 'font_path' in os.environ:
        prop = mfm.FontProperties(fname=os.environ['font_path'])
    else:
        prop = ''
    fig, ax = plt.subplots()
    fig.set_size_inches(45, 15)
    ax.pcolormesh(attn)
    ax.set_title(info)
    for y in range(len(y_script)):
        ax.text(x=-1, y=y + 0.25, s=y_script[y], fontproperties=prop, fontsize='xx-small')
        for x in range(len(x_script[y])):
            ax.text(x=x + 0.01, y=y + 0.25, s=x_script[y][x], fontproperties=prop, fontsize='xx-small')
    fig.savefig(path + '_%d_%d.pdf' % (k, head), dpi=150)
    plt.close(fig)

def plot_attn_with_label(attn, x_script, y_script, path, y_length=None, do_plot=False):
    if not do_plot:
        pickle.dump({'attn': attn, 'x_script': x_script, 'y_script': y_script, 'y_length': y_length},
                    open(path + '.pickle', 'wb'))
        return
    y_script = '>' + y_script + '<'
    for k, layer_attn in enumerate(attn):
        if y_length is not None:
            layer_attn = layer_attn[:, :, :y_length]
        for head in range(layer_attn.shape[0]):
            plot_single_attn_with_label(x_script, y_script, path, layer_attn[head].T, k, head)

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []

class LookupWindow():
    def __init__(self, name, reduction='avg'):
        self.name = name
        self.values = defaultdict(list)
        self.reduction = reduction

    def update(self, keys, values):
        for i in range(len(keys)):
            if values[i] is None:
                continue
            self.values[keys[i]].append(values[i])

    def clear(self):
        self.values = defaultdict(list)

    def summary(self):
        results = []
        if self.reduction == 'total':
            total = sum([sum(v) for v in self.values.values()])
        for key in self.values:
            v = sum(self.values[key])
            if self.reduction == 'sum':
                v = v
            elif self.reduction == 'total':
                v = v / total
            else:
                v = v / len(self.values[key])
            if key != '':
                key = '/' + key
            results.append((self.name + key, v))
        return results

def calculate_mse_dtw(preds, pred_lengths, targets, target_lengths): # [B, T, M]
    results = []
    for i in range(len(preds)):
        x = preds[i, :pred_lengths[i]]
        y = targets[i, :target_lengths[i]]
        voiced = np.where(np.max(x, axis=-1) > 0)
        x = x[voiced]
        voiced = np.where(np.max(y, axis=-1) > 0)
        y = y[voiced]
        if len(x) == 0 or len(y) == 0:
            results.append(None)
            continue

        distance, path = fastdtw(x, y, dist=euclidean)
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x = x[pathx]
        y = y[pathy]
        results.append(np.square(x - y).mean())
    return results