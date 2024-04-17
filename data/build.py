# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

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
