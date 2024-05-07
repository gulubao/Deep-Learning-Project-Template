import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union

# -----------------------------------------------------------------------------------
# All
# -----------------------------------------------------------------------------------
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

# -----------------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------------
def do_resume(checkpoint_path, model, optimizer, args):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    args.logger.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    return model, optimizer, start_epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

