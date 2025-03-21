import argparse

import torch

from data_loader import load_mnist
from utils import get_model


def args_parse():
    p = argparse.ArgumentParser()

    p.add_argument("--weight_fn", required=True, help="model_weight_file_path")

    return p.parse_args()


def load(fn, device):
    d = torch.load(fn, map_location=device, weights_only=False)

    return d["model"], d["config"]


def test(model, x, y):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print(f"Accuracy: {accuracy:.4f}")


def main(config):
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load best model weight and config
    model_weight, train_config = load(config.weight_fn, device)

    # load test data
    x, y = load_mnist(is_train=False, flatten=True if train_config.model == "fc" else False)

    input_size = int(x.size(-1))
    output_size = int(max(y)) + 1

    # set model
    model = get_model(train_config).to(device)
    model.load_state_dict(model_weight)

    # test
    test(model, x.to(device), y.to(device))


if __name__ == "__main__":
    conf = args_parse()
    main(conf)