import importlib
import os

import torch


def dynamic_load(model_class_fqn):
    module_name = ".".join(model_class_fqn.split(".")[:-1])
    class_name = model_class_fqn.split(".")[-1]
    mod = importlib.import_module(module_name)
    class_obj = getattr(mod, class_name)
    return class_obj


def load_optimizer(parameters, config):
    class_obj = dynamic_load(config["optimizer"]["fqdn"])
    optimizer = class_obj(parameters, **config["optimizer"]["kwargs"])
    return optimizer


def load_scheduler(optimizer, config, auto_resume=False, last_iters=100):
    class_obj = dynamic_load(config["scheduler"]["fqdn"])
    kwargs = dict(config["scheduler"]["kwargs"])
    if auto_resume and last_iters > 0 and False:
        kwargs.update({"last_epoch": last_iters})
    scheduler = class_obj(optimizer, **kwargs)
    return scheduler


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 / 1024


def is_kaggle():
    return os.path.abspath(os.curdir).startswith("/kaggle")
