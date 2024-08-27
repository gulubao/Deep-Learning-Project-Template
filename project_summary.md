# train.sh

```sh
# -- environment setup --# cd /home/gulu/code/research/human_adaption_predict conda activate human_behaviour_prediction free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m # -- remove cache --# BASE_DIR="~/code/reproduce/...
```

# README.md


# .gitignore

```
__pycache__ .ipynb_checkpoints .idea .pypirc dist torchlib.egg-info
```

# utils/__init__.py

```py

```

# tools/train_net.py

```py


import argparse
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

work_dir = Path(__file__).resolve().parent.parent
sys.path.append(work_dir)

from config.defaults import default_parser
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model, create_loss
from solver import make_optimizer, make_scheduler
from engine.utils import random_seed, do_resume

from matplotlib import font_manager as fm, pyplot as plt
font_path = '/mnt/c/Windows/Fonts/calibri.ttf'
fm.fontManager.addfont(font_path)
plt.rc('font', family='Calibri')

def train(args):
    model = build_model(
        args=args,
        device=args.device,
        jit=args.jit,
        output_dict=args.output_dict
    )
    loss = create_loss(args)
    train_loader = make_data_loader(args, is_train=True)
    val_loader = make_data_loader(args, is_train=False)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer, train_loader)
    writer = SummaryWriter(log_dir=args.output_dir)

    if args.resume is not None:
        model, optimizer, start_epoch = do_resume(checkpoint_path=args.resume, model=model, optimizer=optimizer, args=args)
    else:
        start_epoch = 0

    args.logger.info("model:\n{}".format(model))
    args.logger.info("loss:\n{}".format(loss))
    args.logger.info("optimizer: {}".format(optimizer))
    args.logger.info("scheduler: {}".format(scheduler))
    args.logger.info("train_loader batch size: {}".format(train_loader.batch_size))
    args.logger.info("train_loader num_samples: {}".format(train_loader.num_samples))
    args.logger.info("train_loader num_batches: {}".format(train_loader.num_batches))
    args.logger.info("val_loader num_samples: {}".format(val_loader.num_samples))
    args.logger.info("val_loader num_batches: {}".format(val_loader.num_batches)) 
    args.logger.info("val_loader batch size: {}".format(val_loader.batch_size))
    args.logger.info("household sample shape: {}".format(train_loader.dataset.features_household.shape))
    args.logger.info("unit sample shape: {}".format(train_loader.dataset.features_unit.shape))

    do_train(
        args = args,
        start_epoch=start_epoch,
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        optimizer = optimizer,
        scheduler = scheduler,
        loss_fn = loss,
        tb_writer=writer
    )


def main(args):
    random_seed(seed=args.seed, rank=0)
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    train(args)


if __name__ == '__main__':
    args = default_parser()
    if args.debug:
        import debugpy
        try:
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass

    main(args)

```

# tools/test_net.py

```py

from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import os
import sys
from os import mkdir
from pathlib import Path
work_dir = Path(__file__).resolve().parent.parent
sys.path.append(work_dir)

from config.defaults import default_parser
from data import make_data_loader
from engine.inference import inference
from modeling import build_model

from matplotlib import font_manager as fm, pyplot as plt
font_path = '/mnt/c/Windows/Fonts/calibri.ttf'
fm.fontManager.addfont(font_path)
plt.rc('font', family='Calibri')

def main():
    model = build_model(args)
    model.load_state_dict(torch.load(args.TEST.WEIGHT))
    val_loader = make_data_loader(args, is_train=False)
    inference(args, model, val_loader)

if __name__ == '__main__':
    args = default_parser()
    if args.debug:
        import debugpy
        try:
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass
    main()

```

# tools/__init__.py

```py


```

# tests/test_data_samplers.py

```py


import sys
import unittest

sys.path.append('.')
from config.defaults import _C as args
from data.transforms import build_transforms
from data.build import build_dataset
from solver.build import make_optimizer
from modeling import build_model


class TestDataSet(unittest.TestCase):
    def test_optimzier(self):
        model = build_model(args)
        optimizer = make_optimizer(args, model)
        from IPython import embed;
        embed()

    def test_args(self):
        args.merge_from_file('configs/train_mnist_softmax.yml')
        from IPython import embed;
        embed()

    def test_dataset(self):
        train_transform = build_transforms(args)
        val_transform = build_transforms(args, False)
        train_set = build_dataset(train_transform)
        val_test = build_dataset(val_transform, False)
        from IPython import embed;
        embed()


if __name__ == '__main__':
    unittest.main()

```

# tests/__init__.py

```py


```

# solver/lr_scheduler.py

```py


# encoding: utf-8
"""
Copied from open_clip/src/training/scheduler.py
"""
import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

```

# solver/build.py

```py


import torch
from solver.lr_scheduler import cosine_lr, const_lr, const_lr_cooldown

def make_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.SOLVER.BASE_LR
        weight_decay = args.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = args.SOLVER.BASE_LR * args.SOLVER.BIAS_LR_FACTOR
            weight_decay = args.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, args.SOLVER.OPTIMIZER_NAME)(params, momentum=args.SOLVER.MOMENTUM)
    return optimizer


def make_scheduler(args, optimizer, train_loader):
    scheduler = None
    if args.lr_scheduler == "none":
        return scheduler
    
    total_steps = (train_loader.num_batches // args.accum_freq) * args.epochs
    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_steps, total_steps)
    elif args.lr_scheduler == "const":
        scheduler = const_lr(optimizer, args.lr, args.warmup_steps, total_steps)
    elif args.lr_scheduler == "const-cooldown":
        assert args.epochs_cooldown is not None,\
            "Please specify the number of cooldown epochs for this lr schedule."
        cooldown_steps = (train_loader.num_batches // args.accum_freq) * args.epochs_cooldown
        scheduler = const_lr_cooldown(
            optimizer, args.lr, args.warmup_steps, total_steps,
            cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
    else:
        args.logging.error(
            f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
        exit(1)
    return scheduler
```

# solver/__init__.py

```py


from .build import make_optimizer, make_scheduler

```

# modeling/model.py

```py


import numpy as np
import torch.nn.functional as F
from torch import nn

from layers.conv_layer import conv3x3


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


```

# modeling/loss.py

```py

import torch
import torch.nn as nn
from torch.nn import functional as F

class ClipLoss(nn.Module):
...
    
```

# modeling/__init__.py

```py

from typing import Any, Dict, Optional, Tuple, Union
import torch
from .model import ResNet18
from .loss import ClipLoss


def build_model(
        args,
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        output_dict: bool = False
    ):
    model = ResNet18(args.NUM_CLASSES)
    model.to(device=device)
    if jit:
        model = torch.jit.script(model)
    if args.torchcompile:
        model = torch.compile(model, mode=args.torchcompile_mode)
    return model

def create_loss(args):
    """references/open_clip/src/open_clip/factory.py"""
    loss_fn = ClipLoss(cache_labels=args.cache_labels)
    return loss_fn
```

# layers/conv_layer.py

```py


from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

```

# engine/utils.py

```py
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


```

# engine/trainer.py

```py

import math
import torch
import os
import time
from .inference import evaluate
from .utils import unwrap_model, AverageMeter

def do_train(
        args,
        start_epoch,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        tb_writer
):

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    for epoch in range(start_epoch, args.max_epochs):
        args.logger.info(f'Start epoch {epoch}')
        train_one_epoch(model, train_loader, loss_fn, epoch, optimizer, scheduler, args, tb_writer=tb_writer)
        completed_epoch = epoch + 1

        # evaluate
        if (completed_epoch == args.max_epochs or ((completed_epoch % args.evaluate_period) == 0) or completed_epoch==1) and (val_loader is not None):
            evaluate(model, train_loader, completed_epoch, args, tb_writer=tb_writer, data_type = "train")
            evaluate(model, val_loader, completed_epoch, args, tb_writer=tb_writer, data_type = "test")

        # Saving checkpoints.
        if completed_epoch == args.max_epochs or ((completed_epoch % args.checkpoint_period) == 0) or completed_epoch==1:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint_dict, args.checkpoint_path / f"epoch_{completed_epoch}.pt")
            # Count the number of files matching args.checkpoint_path / f"epoch_{completed_epoch}.pt".
            checkpoint_files = list(args.checkpoint_path.glob("epoch_*.pt"))
            # If the number of matching files exceeds 3, delete the one with the smallest completed_epoch until there are exactly 3 files.
            if len(checkpoint_files) > 3:
                sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
                for file_to_delete in sorted_checkpoints[:-3]:
                    file_to_delete.unlink()


def train_one_epoch(model, dataloader, loss, epoch, optimizer, scheduler, args, tb_writer=None):
    """open_clip/src/training/train.py --> def train_one_epoch"""
...
```

# engine/inference.py

```py

import logging

def inference(
        args,
        model,
        val_loader
):
    device = args.MODEL.DEVICE

    logger = logging.getLogger("template_model.inference")
    logger.info("Start inferencing")

    evaluate()


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator

```

# data/collate_batch.py

```py


```

# data/build.py

```py


from torch.utils import data

from .datasets.mnist import MNIST
from .transforms import build_transforms


def build_dataset(transforms, is_train=True):
    datasets = MNIST(root='./', train=is_train, transform=transforms, download=True)
    return datasets


def make_data_loader(args, is_train=True):
    if is_train:
        batch_size = args.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = args.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(args, is_train)
    datasets = build_dataset(transforms, is_train)

    num_workers = args.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader

```

# data/__init__.py

```py


from .build import make_data_loader

```

# .vscode/launch.json

```json
{ // 使用 IntelliSense 了解相关属性。 // 悬停以查看现有属性的描述。 // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387 "version": "0.2.0", "configurations": [ { "name": "sh_file_debug", "type": "debugpy", "request": "attach", "connect": { "host": "localhost", "port": 9501 // sudo lsof -i :9501 # 找出占用9501端口的进程 }, "justMyCode": false, "pathMappings": [ // Debug only into specific packages { "localRoot": "${workspaceFolder}", "remoteRoot": "${workspaceFolder}" },{ "localRoot": "~/miniconda3/envs/tf/lib/python3.11/site-packages/transformers", "remoteRoot": "~/miniconda3/envs/tf/lib/python3.11/site-packages/transformers" },{ "localRoot": "~/miniconda3/envs/tf/lib/python3.11/site-packages/datasets", "remoteRoot": "~/miniconda3/envs/tf/lib/python3.11/site-packages/datasets" } ] } ] }
```

# config/defaults.py

```py
import argparse
import logging
import textwrap
import os
import sys
from pathlib import Path
from io import StringIO

def default_parser():
    parser = argparse.ArgumentParser(description="Default Configuration")

    # -----------------------------------------------------------------------------
    # Experiment Parameters
    # -----------------------------------------------------------------------------
    parser.add_argument("seed", default=42, help="Seed for random number generator", type=int)
    parser.add_argument("device", default="cuda", help="Device to use for training", type=str)
    parser.add_argument("experiment_name", default="experiment_debug", help="Experiment name", type=str)
    parser.add_argument("output_dir", default="logs/experiment_debug", help="Output directory", type=str)
    parser.add_argument("debug", default=False, help="Debug mode", type=bool)

    # -----------------------------------------------------------------------------
    # Input Parameters
    # -----------------------------------------------------------------------------
    parser.add_argument("input_dir", default="", help="Path to the input directory", type=str)
    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    parser.add_argument("num_workers", default=8, help="Number of data loading threads", type=int)
    parser.add_argument("batch_size", default=16, help="Batch size", type=int)

    # -----------------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------------
    parser.add_argument("optimizer_name", default="AdamWScheduleFree", help="Optimizer name", type=str)
    parser.add_argument("base_lr", default=0.001, help="Base learning rate", type=float)
    parser.add_argument("bias_lr_factor", default=2, help="Bias learning rate factor", type=float)
    parser.add_argument("momentum", default=0.9, help="Momentum", type=float)
    parser.add_argument("weight_decay", default=0.0005, help="Weight decay", type=float)
    parser.add_argument("weight_decay_bias", default=0, help="Weight decay bias", type=float)

    parser.add_argument("warmup_factor", default=1.0 / 3, help="Warmup factor", type=float)

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    parser.add_argument("max_epochs", default=50, help="Number of training epochs", type=int)
    parser.add_argument("checkpoint_period", default=10, help="Checkpoint period", type=int)
    parser.add_argument("log_period", default=100, help="Log period", type=int)

    # -----------------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------------
    parser.add_argument("checkpoint", default="", help="Path to the checkpoint", type=str)
    parser.add_argument("inference_batch_size", default=16, help="Inference batch size", type=int)

    # -----------------------------------------------------------------------------
    # Update parser
    # -----------------------------------------------------------------------------
    args = parser.parse_args()
    args.output_dir = Path("logs").resolve() / args.experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.world_size = num_gpus
    # --- other configurations --- # 


    # --- log configurations --- #
    args.logger = setup_logger("H-U Match ML:", args.output_dir, 0)
    log_args_in_chunks(args, N=4, logger=args.logger)
    return args
    
    


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.log"), mode='a+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def log_args_in_chunks(args, N=4, logger=None):
    """
    将参数以每行N个参数的方式组织成一个字符串，然后一次性记录日志。
    
    参数:
    args (argparse.Namespace): 要记录的参数。
    N (int): 每行参数的数量。
    logger (logging.Logger): 用于记录日志的Logger对象。
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 如果args不是字典，将其转换为字典
    if not isinstance(args, dict):
        args = vars(args)
    
    # 将args字典转换为字符串列表
    args_list = ["{}={}".format(k, v) for k, v in args.items()]
    
    # 将列表分割成大小为N的块
    chunks = [args_list[i:i + N] for i in range(0, len(args_list), N)]
    
    # 使用StringIO来构建最终的日志消息
    log_message = StringIO()
    log_message.write("Running with config:\n")
    
    for chunk in chunks:
        chunk_str = ", ".join(chunk)
        wrapped_lines = textwrap.wrap(chunk_str, width=120)
        log_message.write("\n\t".join(wrapped_lines) + "\n")
    
    # 一次性记录整个日志消息
    logger.info("{}".format(log_message.getvalue().strip()))
```

# config/__init__.py

```py

```

# data/transforms/transforms.py

```py


import math
import random


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

```

# data/transforms/build.py

```py


import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(args, is_train=True):
    normalize_transform = T.Normalize(mean=args.INPUT.PIXEL_MEAN, std=args.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.RandomResizedCrop(size=args.INPUT.SIZE_TRAIN,
                                scale=(args.INPUT.MIN_SCALE_TRAIN, args.INPUT.MAX_SCALE_TRAIN)),
            T.RandomHorizontalFlip(p=args.INPUT.PROB),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=args.INPUT.PROB, mean=args.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(args.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

```

# data/transforms/__init__.py

```py


from .build import build_transforms

```

# data/datasets/mnist.py

```py


from torchvision.datasets import MNIST


```

# data/datasets/__init__.py

```py



```

