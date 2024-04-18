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
import os
import sys
from os import mkdir
from pathlib import Path
work_dir = Path(__file__).resolve().parent.parent
sys.path.append(work_dir)

from config.defaults import default_parser
from config.extra_config import extra_config
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("template_model", args.output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    logger.info("Running with config:\n{}".format(args))

    model = build_model(args)
    model.load_state_dict(torch.load(args.TEST.WEIGHT))
    val_loader = make_data_loader(args, is_train=False)

    inference(args, model, val_loader)

if __name__ == '__main__':
    main()
