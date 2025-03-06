import torch
import torch.nn as nn


class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, output_size)
        )
        self.act = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.act(self.layers(x))
        # |y| = (batch_size, output_size)

        return y