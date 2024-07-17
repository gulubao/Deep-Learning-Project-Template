import argparse
import logging
import textwrap
import os
import sys
from pathlib import Path


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
    args.logger = setup_logger("H-U Match ML:", args.output_dir, 0)
    args.logger.info("Running with config:")
    log_args_in_chunks(args, N=4, logger=args.logger)
    return args
    
    


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.log"), mode='a+')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def log_args_in_chunks(args, N=4, logger=None):
    """
    Logs the args in chunks of N parameters per line.
    
    Parameters:
    args (argparse.Namespace): The arguments to be logged.
    N (int): Number of parameters per line.
    logger (logging.Logger): Logger object to use for logging.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Convert args to a dictionary if it's not already
    if not isinstance(args, dict):
        args = vars(args)
    
    # Convert the args dictionary to a list of strings
    args_list = ["{}={}".format(k, v) for k, v in args.items()]
    
    # Split the list into chunks of size N
    chunks = [args_list[i:i + N] for i in range(0, len(args_list), N)]
    
    # Format each chunk and log it
    for chunk in chunks:
        logger.info("\n".join(textwrap.wrap(", ".join(chunk), width=120)))