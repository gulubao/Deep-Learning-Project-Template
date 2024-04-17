from pathlib import Path
import yaml

def setup_output_dir(args):
    output_dir_name = args.experiment_name
    output_dir = Path("logs").resolve() / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def merge_from_yaml(args):
    with open(args.config_file, 'r') as f:
        yaml_args = yaml.safe_load(f)
    for k, v in yaml_args.items():
        setattr(args, k, v)
    return args


def merge_additional_args(args, default_args):
    """
    Merges additional arguments parsed by argparse with default arguments.

    Parameters:
    args (tuple): A tuple returned by argparse.ArgumentParser.parse_known_args(),
                  where args[0] is a Namespace of known arguments and args[1] is a list of remaining arguments.
    default_args (dict): A dictionary of default arguments.

    Returns:
    dict: A dictionary containing the merged arguments.
    """
    # Convert the Namespace of known arguments to a dictionary
    known_args_dict = vars(args[0])
    
    # Initialize a new dictionary to store the merged arguments
    merged_args = {}
    
    # Update the merged_args dictionary with default arguments
    merged_args.update(default_args)
    
    # Update the merged_args dictionary with known arguments, overriding any defaults if necessary
    merged_args.update(known_args_dict)
    
    return merged_args