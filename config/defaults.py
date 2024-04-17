import argparse

def default_parser():
    parser = argparse.ArgumentParser(description="Default Configuration")

    # -----------------------------------------------------------------------------
    # Experiment Parameters
    # -----------------------------------------------------------------------------
    parser.add_argument("seed", default=42, help="Seed for random number generator", type=int)
    parser.add_argument("device", default="cuda", help="Device to use for training", type=str)
    parser.add_argument("experiment_name", default="experiment_debug", help="Experiment name", type=str)
    parser.add_argument("checkpoint_path", default="logs/experiment_debug/checkpoints", help="Path to the checkpoint", type=str)
    parser.add_argument("output_dir", default="logs/experiment_debug", help="Output directory", type=str)
    parser.add_argument("debug", default=True, help="Debug mode", type=bool)
    parser.add_argument("config_file", default="", help="Path to the config file", type=str)  

    # -----------------------------------------------------------------------------
    # Input Parameters
    # -----------------------------------------------------------------------------
    parser.add_argument("input_dir", default="", help="Path to the input directory", type=str)
    parser.add_argument("input_raw_file", default="dataset/row_data/psam_h10.csv", help="Raw csv household-unit data file", type=str)
    parser.add_argument("interest_column_file", default="dataset/row_data/interest_variables.xlsx", help="File containing the interested columns", type=str)
    parser.add_argument("tmp_dir", default="dataset/tmp", help="File containing the interested columns", type=str)
    parser.add_argument("input_train_house_file", default="dataset/processed_data/house_train.csv", help="Training house data file", type=str)
    parser.add_argument("input_train_unit_file", default="dataset/processed_data/unit_train.csv", help="Training unit data file", type=str)
    parser.add_argument("input_test_house_file", default="dataset/processed_data/house_test.csv", help="Testing house data file", type=str)
    parser.add_argument("input_test_unit_file", default="dataset/processed_data/unit_test.csv", help="Testing unit data file", type=str)

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
    parser.add_argument("warmup_iters", default=500, help="Warmup iterations", type=int)
    parser.add_argument("warmup_method", default="linear", help="Warmup method", type=str)

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
    args = parser.parse_args()
    return args
    
    