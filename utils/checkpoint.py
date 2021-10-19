import os
from collections import OrderedDict
import torch
import glob
import logging


def find_ckpt(base_dir):
    max_step = 0
    result = None
    for f in glob.iglob(os.path.join(base_dir, 'model.ckpt-*')):
        step = int(f.split('-')[-1])
        if step > max_step:
            result = f
            max_step = step
    return result


def save_model(model_dir, model=None, optim=None, sched=None, step=None):
    state_dict = {}
    if model:
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        state_dict['model'] = model_dict
    if optim:
        state_dict['optim'] = optim.state_dict()
    if sched:
        state_dict['sched'] = sched.state_dict()
    if step:
        state_dict['step'] = step
        model_dir = os.path.join(model_dir, 'model.ckpt-%d' % step)
    torch.save(state_dict, model_dir)

def build_dict_update(base_dict, updated_dict):
    result_dict = {}
    used_keys = set()
    for key in base_dict:
        if key in updated_dict:
            if base_dict[key].shape == updated_dict[key].shape:
                result_dict[key] = updated_dict[key]
                used_keys.add(key)
            else:
                logging.warn("Parameter shape mismatch: %s; %s to %s" % (
                    key, base_dict[key].shape, updated_dict[key].shape))
        else:
            logging.warn("Parameter not found in updated: %s; %s" % (key, base_dict[key].shape))
    for key in updated_dict:
        if key not in used_keys:
            logging.warn("Parameter not found in base: %s; %s" % (key, updated_dict[key].shape))
    return result_dict

def load_model(model_path, model=None, optim=None, sched=None, map_location={}, strict=False):
    state_dict = torch.load(model_path, map_location=map_location)
    model_changed = False
    if 'model' in state_dict and model:
        current_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        updated_dict = build_dict_update(current_dict, state_dict['model'])
        if hasattr(model, 'module'):
            model.module.load_state_dict(updated_dict, strict=strict)
        else:
            model.load_state_dict(updated_dict, strict=strict)
        if set(current_dict.keys()) != set(state_dict['model'].keys()):
            model_changed = True
    if 'optim' in state_dict and optim and not model_changed:
        optim.load_state_dict(state_dict['optim'])
    if 'step' in state_dict:
        step = state_dict['step']
    else:
        step = None
    if 'sched' in state_dict and sched:
        sched.load_state_dict(state_dict['sched'])
        if step:
            if step != sched.last_epoch:
                logging.warn("Step=%d, while in sched step=%d" % (step, sched.last_epoch))
        else:
            step = sched.last_epoch
    return step
