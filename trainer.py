import numpy as np
from copy import deepcopy

import torch


class Trainer:
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, train_loader):
        # set train mode
        self.model.train()

        total_loss = 0

        for x_i, y_i in train_loader:
            # forward
            y_hat_i = self.model(x_i)

            # calculate train loss_i value
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # initialize gradient
            self.optimizer.zero_grad()

            # backpropagation
            loss_i.backward()

            # gradient descent
            self.optimizer.step()

            total_loss += float(loss_i)

        return total_loss / len(train_loader)

    def _validate(self, valid_loader):
        # set evaluation model
        self.model.eval()

        # turn off grad mode
        with torch.no_grad():
            total_loss = 0

            for x_i, y_i in valid_loader:
                # forward
                y_hat_i = self.model(x_i)

                # calculate valid loss value
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                total_loss += float(loss_i)

            return total_loss / len(valid_loader)

    def train(self, train_loader, valid_loader, config):
        lowest_loss = np.inf
        lowest_epoch = np.inf
        best_model = None

        for epoch_idx in range(config.n_epochs):
            train_loss = self._train(train_loader)
            valid_loss = self._validate(valid_loader)

            print(f"Epoch({epoch_idx + 1}/{config.n_epochs}) - train_loss={train_loss:.4e} valid_loss={valid_loss:.4e} lowest_loss={lowest_loss:.4e}")

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                lowest_epoch = epoch_idx
                best_model = deepcopy(self.model.state_dict())
            else:
                if config.early_stop > 0 and lowest_epoch + config.early_stop < epoch_idx + 1:
                    print(f"There is no improvement during last {config.early_stop} epochs.")
                    break

        print(f"The best valid loss from epoch {lowest_epoch}: {lowest_loss:.4e}")

        # restore best model weights
        self.model.load_state_dict(best_model)