import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model import FullyConnectedClassifier
from trainer import Trainer
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
    # train info log
    info = "[Train config information]\n"

    # set device
    device = torch.device("cpu") if config.gpu_id < 0 else torch.device(f"cuda:{config.gpu_id}")
    info += f"device: {device}\n"

    # load and split data into train/valid data
    x, y = load_data(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), device, train_ratio=config.train_ratio)
    info += "[train/valid data shape]\n"
    info += f"train: {x[0].shape}/{y[0].shape}\n"
    info += f"valid: {x[1].shape}/{y[1].shape}\n"

    input_size = int(x[0].size(-1))
    output_size = int(max(y[0])) + 1

    # set model, optimizer, criterion
    model = FullyConnectedClassifier(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    info += f"[model]\n{str(model)}\n"
    info += f"[optimizer]\n{str(optimizer)}\n"
    info += f"criterion : {str(crit)}"

    # printing train information
    print(info)

    # set trainer
    trainer = Trainer(model, optimizer, crit)

    # train model
    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # save best model weights
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': trainer.optimizer.state_dict(),
        'config': config,
    }, config.weight_fn)


if __name__ == "__main__":
    args = args_parse()
    conf = load_config(args.config_path)
    main(conf)