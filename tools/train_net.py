# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

import argparse
from typing import Any, Dict, List, Tuple, Union, Optional
import torch.nn.functional as F
import os
import sys
from pathlib import Path
work_dir = Path(__file__).resolve().parent.parent
sys.path.append(work_dir)

from config.defaults import default_parser
from config.extra_config import extra_config
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer

from utils.logger import setup_logger


def train(args):
    model = build_model(args)
    device = args.device

    optimizer = make_optimizer(args, model)
    scheduler = None

    arguments = {}

    train_loader = make_data_loader(args, is_train=True)
    val_loader = make_data_loader(args, is_train=False)

    do_train(
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        None,
        F.cross_entropy,
    )


def main():
    args = default_parser()
    # Update duplicate properties in args using extra_config.
    for key, value in vars(extra_config).items():
        setattr(args, key, value)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("template_model", args.output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    logger.info("Running with config:\n{}".format(args))

    train(args)


if __name__ == '__main__':
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9501))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass

    main()
