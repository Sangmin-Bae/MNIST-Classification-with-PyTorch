import argparse
import yaml

import torch


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
    print(f"device - {device}\n")

    # load data

if __name__ == "__main__":
    args = arguments_parser()
    conf = load_config(args.config_path)
    main(conf)
