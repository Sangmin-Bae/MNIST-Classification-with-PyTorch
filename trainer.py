import numpy as np
from copy import deepcopy

import torch


class Trainer:
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    @staticmethod
    def _batchify(x, y, batch_size, random_split=True):
        # Shuffling
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        # Split data based batch_size
        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    def _train(self, x, y, config):
        self.model.train()

        x, y = self._batchify(x, y, config.batch_size, random_split=True)
        total_loss = 0

        for idx, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        self.model.eval()

        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for idx, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        lowest_epoch = np.inf
        best_model = None

        for epoch_idx in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            if (epoch_idx + 1) % config.print_interval == 0:
                print(f"Epoch({epoch_idx + 1}/{config.n_epochs}): train_loss={train_loss:.4e} valid_loss={valid_loss:.4e} lowest_loss={lowest_loss:.4e}")

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                lowest_epoch = epoch_idx
                best_model = deepcopy(self.model.state_dict())
            else:
                if config.early_stop > 0 and lowest_epoch + config.early_stop < epoch_idx + 1:
                    print(f"There is no improvement during last {config.early_stop} epochs.")
                    break

        print(f"The best validation loss from epoch {lowest_epoch + 1}: {lowest_epoch:.4e}")
        # restore best model weight
        self.model.load_state_dict(best_model)