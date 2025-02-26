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