# encoding: utf-8
"""
Template follow:
    https://github.com/L1aoXingyu/Deep-Learning-Project-Template
    https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch
    https://github.com/yuanzhoulvpi2017/vscode_debug_transformers
"""
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import os
import sys
from os import mkdir
from pathlib import Path
work_dir = Path(__file__).resolve().parent.parent
sys.path.append(work_dir)

from config.defaults import default_parser
from data import make_data_loader
from engine.inference import inference
from modeling import build_model

from matplotlib import font_manager as fm, pyplot as plt
font_path = '/mnt/c/Windows/Fonts/calibri.ttf'
fm.fontManager.addfont(font_path)
plt.rc('font', family='Calibri')

def main():
    model = build_model(args)
    model.load_state_dict(torch.load(args.TEST.WEIGHT))
    val_loader = make_data_loader(args, is_train=False)
    inference(args, model, val_loader)

if __name__ == '__main__':
    args = default_parser()
    if args.debug:
        import debugpy
        try:
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(("localhost", 9501))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
        except Exception as e:
            pass
    main()
