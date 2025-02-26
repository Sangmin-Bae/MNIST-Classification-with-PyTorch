import argparse
import yaml

import torch

from utils import load_data, split_data


def arguments_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--config_path", required=True, help="config_file_path[type: yaml]")

    return p.parse_args()


def load_config(path):
    # load config.yaml and convert argparse.Namespace object
    with open(path, 'r') as f:
        y = yaml.safe_load(f)
    return argparse.Namespace(**y)


def main(config):
    # set device
    device = torch.device("cpu") if config.gpu_id < 0 and not torch.cuda.is_available() else torch.device(f"cuda:{config.gpu_id}")
    print(f"device - {device}")

    # load data
    x, y = load_data(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), device, config.train_ratio)
    print(f"Train - {x[0].shape} / {y[0].shape}")
    print(f"Valid - {x[1].shape} / {y[1].shape}")

if __name__ == "__main__":
    args = arguments_parser()
    conf = load_config(args.config_path)
    main(conf)
