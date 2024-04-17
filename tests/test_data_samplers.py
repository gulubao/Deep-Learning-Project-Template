# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""

import sys
import unittest

sys.path.append('.')
from config.defaults import _C as args
from data.transforms import build_transforms
from data.build import build_dataset
from solver.build import make_optimizer
from modeling import build_model


class TestDataSet(unittest.TestCase):
    def test_optimzier(self):
        model = build_model(args)
        optimizer = make_optimizer(args, model)
        from IPython import embed;
        embed()

    def test_args(self):
        args.merge_from_file('configs/train_mnist_softmax.yml')
        from IPython import embed;
        embed()

    def test_dataset(self):
        train_transform = build_transforms(args)
        val_transform = build_transforms(args, False)
        train_set = build_dataset(train_transform)
        val_test = build_dataset(val_transform, False)
        from IPython import embed;
        embed()


if __name__ == '__main__':
    unittest.main()
