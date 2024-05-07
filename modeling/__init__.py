# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""
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