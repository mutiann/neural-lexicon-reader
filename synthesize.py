import torch
import time
import copy
from hyperparams import hparams as hp
import numpy as np
import logging
import tqdm
import traceback
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.infolog import plot_attn, plot_mel, plot_attn_with_label
from utils.audio import mel2wav, save_wav, trim_silence_intervals

def eval_batch(model_eval, data, use_bar=True, bar_interval=10):
    with torch.no_grad():
        tic = time.time()
        batch = copy.copy(data)
        device = batch['inputs'].device
        batch_size = batch['inputs'].shape[0]
        target_lengths = torch.ones([batch_size], dtype=torch.int32, device=device)
        finished = torch.zeros([batch_size], dtype=torch.bool, device=device)
        mels = torch.zeros([batch_size, 0, hp.num_mels], dtype=torch.float32, device=device)  # [B, T, M]
        if 'input_spk_ids' not in batch:
            batch['input_spk_ids'] = None
        if 'input_language_vecs' not in batch:
            batch['input_language_vecs'] = None
        enc_outputs = model_eval.encoder(**batch)
        torch.cuda.empty_cache()
        if use_bar:
            bar = tqdm.tqdm()
        while not torch.all(finished) and mels.shape[1] < hp.max_generation_frames:
            try:
                decoder_input = torch.cat([mels, torch.zeros([batch_size, 1, hp.num_mels], device=device)], dim=1)

                mel_bef, stop_logits, align = \
                    model_eval.decoder(enc_outputs, batch['input_lengths'],
                                       decoder_input, target_lengths, leave_one=True)
                stop = stop_logits[:, -1] > 0
                mels = torch.cat([mels, mel_bef[:, -1:]], dim=1)
                finished = torch.logical_or(finished, stop)
                target_lengths = torch.where(finished, target_lengths, target_lengths + 1)

                if mels.shape[1] % bar_interval == 0 and bar_interval != -1:
                    if use_bar:
                        bar.update(bar_interval)
                    else:
                        print(mels.shape[1])
            except:
                traceback.print_exc()
                break

        mel_aft = mels + model_eval.postnet(mels, target_lengths)

        # Evade memory leakage
        for key in ['encdec']:
            for i in range(len(align[key])):
                align[key][i] = align[key][i].cpu().numpy()

        if hasattr(model_eval.encoder, 'alignments'):
            align['knowledge'] = model_eval.encoder.alignments
            for i in range(len(align['knowledge'])):
                align['knowledge'][i] = align['knowledge'][i].cpu().numpy()

        toc = time.time()
        total_length = target_lengths.sum().item()
        logging.info("Time: %.4f, Samples: %d, Length: %d, Max length: %d, Real-time Factor: %.4f" % (
            toc - tic, mels.shape[0], total_length, target_lengths.max().item(),
            (toc - tic) / total_length * 80))

        return {'names': data['names'], 'mel_pre': mels.cpu().numpy(),
                'mel_aft': mel_aft.cpu().numpy(), 'alignments': align,
                'generated_lengths': list(target_lengths.cpu().numpy())}


def save_eval_results_i(name, mel, generated_length, input_length, encdec_align, knowledge_align,
                        context_scripts, input_scripts, output_dir, save_trimmed_wave):
    try:
        mel = mel[:generated_length]
        np.save(os.path.join(output_dir, '%s.npy' % name), mel)
        wav = mel2wav(mel)
        save_wav(wav, os.path.join(output_dir, '%s.wav' % name))
        if save_trimmed_wave:
            wav_trim = trim_silence_intervals(wav)
            save_wav(wav_trim, os.path.join(output_dir, '%s_trim.wav' % name))
        plot_mel(os.path.join(output_dir, '%s_mel.png' % name), mel)

        if encdec_align:
            plot_attn(encdec_align, os.path.join(output_dir, '%s_align.png' % (name)),
                      enc_length=input_length, dec_length=generated_length)
        if knowledge_align:  # [batch, heads, context, inputs]
            plot_attn_with_label(
                knowledge_align, context_scripts, input_scripts,
                os.path.join(output_dir, '%s_knowledge_align' % (name)), y_length=input_length)

    except:
        logging.error('Fail to produce eval output: ' + name)
        logging.error(traceback.format_exc())


def save_eval_results(names, mel_pre, mel_aft, alignments, generated_lengths,
                      output_dir, inputs, save_trimmed_wave=False, n_plot_alignment=None):
    executor = ProcessPoolExecutor(max_workers=4)
    tic = time.time()
    futures = []
    for i in range(len(names)):
        if n_plot_alignment is None or i < n_plot_alignment:
            encdec_align = [a[i].transpose([0, 2, 1]) for a in alignments["encdec"]]
        else:
            encdec_align = None
        if 'knowledge' in alignments:
            knowledge_align = [a[i] for a in alignments["knowledge"]]
            context_scripts = inputs['context_scripts'][i]
            input_scripts = inputs['input_scripts'][i]
        else:
            context_scripts = input_scripts = knowledge_align = None
        futures.append(executor.submit(
            partial(save_eval_results_i, name=names[i], mel=mel_aft[i], generated_length=generated_lengths[i],
                    input_length=inputs['input_lengths'][i], encdec_align=encdec_align, knowledge_align=knowledge_align,
                    context_scripts=context_scripts, input_scripts=input_scripts,
                    output_dir=output_dir, save_trimmed_wave=save_trimmed_wave)))

    [future.result() for future in futures]

    logging.info('[%s] Finished saving evals in %.2f secs: ' %
                 (threading.current_thread().name, time.time() - tic) + str(names))