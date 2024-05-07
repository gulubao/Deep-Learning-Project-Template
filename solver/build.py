# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

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