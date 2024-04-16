# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
"""

from .example_model import ResNet18


def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
