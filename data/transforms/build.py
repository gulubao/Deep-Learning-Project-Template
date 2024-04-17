# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

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
