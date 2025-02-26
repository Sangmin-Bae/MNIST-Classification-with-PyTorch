import argparse

import torch

from model.fcnn_model import FullyConnectedClassifier

from utils import load_data


def arguments_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--weight_path", required=True, help="model_weight_file_path")

    return p.parse_args()


def load(weight_path, device):
    d = torch.load(weight_path, map_location=device, weights_only=False)

    return d["model"], d["config"]


def show_confusion_matrix(y, y_hat):
    import pandas as pd
    from tabulate import tabulate
    from sklearn.metrics import confusion_matrix


    cm = pd.DataFrame(confusion_matrix(y, torch.argmax(y_hat, dim=-1)),
                index=[f"true_{i}" for i in range(10)],
                columns=[f"pred_{i}" for i in range(10)])
    print(f"[Confusion Matrix]\n")
    print(tabulate(cm, headers='keys'))


def test(model, x, y, to_be_shown=True):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print(f"Accuracy: {accuracy:.4f}")

        if to_be_shown:
            show_confusion_matrix(y.to('cpu'), y_hat.to('cpu'))


def main(config):
    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load data
    x, y = load_data(is_train=False, flatten=True)

    input_size = int(x.size(-1))
    output_size = int(max(y)) + 1

    # load model weight
    model_weight_dict, train_config = load(config.weight_path, device)

    # set model
    model = FullyConnectedClassifier(input_size, output_size).to(device)
    model.load_state_dict(model_weight_dict)

    # test
    test(model, x.to(device), y.to(device), to_be_shown=True)


if __name__ == "__main__":
    conf = arguments_parser()
    main(conf)