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

import torch

sys.path.append('.')
from config import args
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    args = default_parser()
    # Update duplicate properties in args using extra_config.
    for key, value in vars(extra_config).items():
        setattr(args, key, value)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        args.merge_from_file(args.config_file)
    args.merge_from_list(args.opts)
    args.freeze()

    output_dir = args.OUTPUT_DIR
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

    model = build_model(args)
    model.load_state_dict(torch.load(args.TEST.WEIGHT))
    val_loader = make_data_loader(args, is_train=False)

    inference(args, model, val_loader)


if __name__ == '__main__':
    main()
