# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

import argparse
from config.defaults import default_parser
from config.extra_config import extra_config
import os
import sys
from os import mkdir
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import torch.nn.functional as F

sys.path.append('.')
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

    if args.config_file != "":
        args.merge_from_file(args.config_file)

    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
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
