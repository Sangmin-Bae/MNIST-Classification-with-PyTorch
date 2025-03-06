import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model import FullyConnectedClassifier
from utils import load_data, split_data


def args_parse():
    p = argparse.ArgumentParser()

    # get config.yaml file path
    p.add_argument("--config_path", required=True, help="config_yaml_file_path[ex) ./config.yaml]")

    return p.parse_args()


def load_config(path):
    """ load train config """
    with open(path) as f:
        y = yaml.safe_load(f)

    return argparse.Namespace(**y)


def main(config):
    # set device
    device = torch.device("cpu") if config.gpu_id < 0 else torch.device(f"cuda:{config.gpu_id}")

    # load and split data into train/valid data
    x, y = load_data(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), device, train_ratio=config.train_ratio)

    input_size = int(x.size(0))
    output_size = int(max(y)) + 1

    # set model, optimizer, criterion
    model = FullyConnectedClassifier(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()


if __name__ == "__main__":
    args = args_parse()
    conf = load_config(args.config_path)
    main(conf)