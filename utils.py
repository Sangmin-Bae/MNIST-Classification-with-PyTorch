import torch

from torchvision import transforms
from torchvision.datasets import MNIST


def load_data(is_train=True, flatten=True):
    """
    load mnist train/test dataset
    """
    # load mnist data
    dataset = MNIST(
        root="./dataset",
        train=is_train,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=True
    )

    x = dataset.data.float() / 255.  # MinMax Scaling
    y = dataset.targets

    # flatten 28x28 image
    # |x| = (60000, 28, 28) -> |x| = (60000, 784)
    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def split_data(x, y, device, train_ratio=0.8):
    """ shuffle and split train/valid data """
    # calculate train_count, valid_count
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # shuffle and split data into train/valid data
    indices = torch.randperm(x.size(0)).to(device)
    x = torch.index_select(x, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)

    return x, y