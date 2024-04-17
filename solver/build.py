# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

import torch


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
