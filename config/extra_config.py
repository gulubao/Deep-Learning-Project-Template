import argparse
__all__ = ['extra_config']

configs = {
    'have_extra_config': False
}
extra_config = argparse.Namespace(**configs)