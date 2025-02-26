import torch


def load_data(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        root="./dataset/mnist",
        train=is_train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    # MinMax Scaling - [0. - 255.] > [0. - 1.]
    x = dataset.data.float() / 255.
    y = dataset.targets

    # Flatten - |x| = (60000, 28, 28) > |x| = (60000, 784)
    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def split_data(x, y, device, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # Shuffle dataset and Split into train/valid set
    indices = torch.randperm(x.size(0)).to(device)
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y