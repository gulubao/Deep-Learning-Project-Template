# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

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
