import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model.fcnn_model import FullyConnectedClassifier

from trainer import Trainer

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
    device = torch.device(f"cuda:{config.gpu_id}") if config.gpu_id >= 0 and torch.cuda.is_available() else torch.device("cpu")
    print(f"device - {device}")

    # load data
    x, y = load_data(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), device, config.train_ratio)
    print(f"Train - {x[0].shape} / {y[0].shape}")
    print(f"Valid - {x[1].shape} / {y[1].shape}")

    input_size = int(x[0].size(-1))
    output_size = int(max(y[0])) + 1

    # set model, optimizer, crit
    model = FullyConnectedClassifier(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()
    print(model)
    print(optimizer)
    print(crit)

    # train
    trainer = Trainer(model, optimizer, crit)
    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # save model weight
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == "__main__":
    args = arguments_parser()
    conf = load_config(args.config_path)
    main(conf)
