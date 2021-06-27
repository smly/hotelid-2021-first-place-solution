from collections import OrderedDict
from pathlib import Path

import torch


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if 'module' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # str.replace() can't be used because of unintended key removal (e.g. se-module)
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(
    path, model, epoch,
    optimizer=None,
    scheduler=None,
    save_arch=False,
    params=None
):
    attributes = {
        'epoch': epoch,
        'state_dict': remove_redundant_keys(model.state_dict()),
    }

    if optimizer is not None:
        attributes['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        attributes["scheduler"] = scheduler.state_dict()
    if save_arch:
        attributes['arch'] = model

    if params is not None:
        attributes['params'] = params

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(attributes, path)
    except TypeError:
        if 'arch' in attributes:
            print(
                'Model architecture will be ignored because the architecture '
                'includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, path)
