import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model import FullyConnectedClassifier
from trainer import Trainer
from data_loader import get_loaders


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

    # load train/valid/test data loader
    train_loader, valid_loader, test_loader = get_loaders(device, config)

    info += f"train: {len(train_loader.dataset)}\n"
    info += f"valid: {len(valid_loader.dataset)}\n"
    info += f"test: {len(test_loader.dataset)}\n"

    # get input/output size
    x, y = next(iter(train_loader))
    input_size = int(x.size(-1))
    output_size = int(max(y)) + 1

    # set model, optimizer, criterion
    model = FullyConnectedClassifier(input_size, output_size, config.use_batch_norm, config.dropout_p).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    info += f"[model]\n{str(model)}\n"
    info += f"[optimizer]\n{str(optimizer)}\n"
    info += f"criterion : {str(crit)}"

    # printing train information
    print(info)

    # set trainer
    trainer = Trainer(config)

    # train model
    trainer.train(model, crit, optimizer, train_loader, valid_loader)


if __name__ == "__main__":
    args = args_parse()
    conf = load_config(args.config_path)
    main(conf)