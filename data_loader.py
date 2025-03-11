import torch

from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    """ CustomDataset Class for MNIST dataset """
    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # flatten 28x28 image -> 784 vector
        # |x| = (28, 28) -> |x| = (784,)
        if self.flatten:
            x = x.view(-1)

        return x, y


def load_mnist(is_train=True):
    """ MNIST train/valid 데이터 불러오기 """
    from torchvision import datasets, transforms

    # load mnist data
    dataset = datasets.MNIST(
        root="./dataset",
        train=is_train,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        download=True
    )

    # MinMax Scaling
    x = dataset.data.float() / 255.
    y = dataset.targets

    return x, y


def split_data(x, y, config):
    """ shuffle and split train/valid data """
    # set train/valid count
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # shuffle and split data into train/valid data
    indices = torch.randperm(x.size(0))
    x = torch.index_select(x, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)

    return x[0], y[0], x[1], y[1]


def get_loaders(config):
    # load MNIST train data
    x, y = load_mnist(is_train=True)

    # shuffle and split data
    train_x, train_y, valid_x, valid_y = split_data(x, y, config)

    # load MNIST test data
    test_x, test_y = load_mnist(is_train=False)

    # set data loaders
    train_loader = DataLoader(
        dataset=MNISTDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=MNISTDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=MNISTDataset(test_x, test_y, flatten=False),
        batch_size=config.batch_size,
        shuffle=True
    )

    return train_loader, valid_loader, test_loader
