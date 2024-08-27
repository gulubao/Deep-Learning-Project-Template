import argparse
import logging
import textwrap
import os
import sys
from pathlib import Path
from io import StringIO
from transformers import set_seed

def default_parser():
    parser = argparse.ArgumentParser(description="Default Configuration")

    # -----------------------------------------------------------------------------
    # Experiment Parameters
    # -----------------------------------------------------------------------------
    parser.add_argument("seed", default=42, help="Seed for random number generator", type=int)
    parser.add_argument("device", default="cuda", help="Device to use for training", type=str)
    parser.add_argument("experiment_name", default="experiment_debug", help="Experiment name", type=str)
    parser.add_argument("output_dir", default="logs/experiment_debug", help="Output directory", type=str)
    parser.add_argument("debug", default=False, help="Debug mode", type=bool)

    # -----------------------------------------------------------------------------
    # Input Parameters
    # -----------------------------------------------------------------------------
    parser.add_argument("input_dir", default="", help="Path to the input directory", type=str)
    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    parser.add_argument("num_workers", default=8, help="Number of data loading threads", type=int)
    parser.add_argument("batch_size", default=16, help="Batch size", type=int)

    # -----------------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------------
    parser.add_argument("optimizer_name", default="AdamWScheduleFree", help="Optimizer name", type=str)
    parser.add_argument("base_lr", default=0.001, help="Base learning rate", type=float)
    parser.add_argument("bias_lr_factor", default=2, help="Bias learning rate factor", type=float)
    parser.add_argument("momentum", default=0.9, help="Momentum", type=float)
    parser.add_argument("weight_decay", default=0.0005, help="Weight decay", type=float)
    parser.add_argument("weight_decay_bias", default=0, help="Weight decay bias", type=float)

    parser.add_argument("warmup_factor", default=1.0 / 3, help="Warmup factor", type=float)

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    parser.add_argument("max_epochs", default=50, help="Number of training epochs", type=int)
    parser.add_argument("checkpoint_period", default=10, help="Checkpoint period", type=int)
    parser.add_argument("log_period", default=100, help="Log period", type=int)

    # -----------------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------------
    parser.add_argument("checkpoint", default="", help="Path to the checkpoint", type=str)
    parser.add_argument("inference_batch_size", default=16, help="Inference batch size", type=int)

    # -----------------------------------------------------------------------------
    # Update parser
    # -----------------------------------------------------------------------------
    args = parser.parse_args()
    args.output_dir = Path("logs").resolve() / args.experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.world_size = num_gpus
    # --- other configurations --- # 


    # --- log configurations --- #
    args.logger = setup_logger("H-U Match ML:", args.output_dir)
    log_args_in_chunks(args, N=4, logger=args.logger)
    set_seed(args.seed)
    return args
    
    
def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.log"), mode='a+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def log_args_in_chunks(args, N=4, logger=None):
    """
    将参数以每行N个参数的方式组织成一个字符串, 然后一次性记录日志。
    每个参数包括其名称和值。
    
    参数:
    args (argparse.Namespace): 要记录的参数。
    N (int): 每行参数的数量。
    logger (logging.Logger): 用于记录日志的Logger对象。
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    args_name = args.__class__.__name__
    if not isinstance(args, dict):
        args = vars(args)
    
    # 将args字典转换为字符串列表，并找出最长的参数字符串
    args_list = []
    max_param_length = 0
    for k, v in args.items():
        param_str = f"{k} = {v}"
        args_list.append(param_str)
        max_param_length = max(max_param_length, len(param_str))
    
    # 使用StringIO来构建最终的日志消息
    log_message = StringIO()
    log_message.write(f"运行配置 - {args_name}:\n")
    
    # 将参数列表分割成大小为N的块
    for i in range(0, len(args_list), N):
        chunk = args_list[i:i+N]
        # 对齐每个参数
        formatted_chunk = [f"{param:<{max_param_length}}" for param in chunk]
        log_message.write("    " + "  ".join(formatted_chunk) + "\n")
    
    # 一次性记录整个日志消息
    logger.info(log_message.getvalue().strip())