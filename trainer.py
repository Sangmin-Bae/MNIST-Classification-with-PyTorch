import numpy as np
from copy import deepcopy

import torch
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.handlers.tqdm_logger import ProgressBar

from utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MyEngine(Engine):
    def __init__(self, func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None
        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = mini_batch
        x, y = x.to(engine.device), y.to(engine.device)

        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y)
        loss.backward()

        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        if engine.config.max_grad > 0:
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad,
                norm_type=2
            )

        engine.optimizer.step()

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "|param|": p_norm,
            "|g_param|": g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }

    @staticmethod
    def attach(train_engine, validate_engine, verbose=VERBOSE_BATCH_WISE):
        def attach_running_average(engine, m_name):
            RunningAverage(output_transform=lambda x: x[m_name]).attach(engine, m_name)

        training_metric_names = ["loss", "accuracy", "|param|", "|g_param|"]

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)


        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print("Epoch {} - |param|={:2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}".format(
                    engine.state.epoch,
                    engine.state.metrics["|param|"],
                    engine.state.metrics["|g_param|"],
                    engine.state.metrics["loss"],
                    engine.state.metrics["accuracy"],
                ))

        validation_metric_names = ["loss", "accuracy"]

        for metric_name in validation_metric_names:
            attach_running_average(validate_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validate_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validate_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print("Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}".format(
                    engine.state.metrics["loss"],
                    engine.state.metrics["accuracy"],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics["loss"])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, config, **kwargs):
        torch.save(
            {
                "model": engine.best_model,
                "config": config,
                **kwargs,
            }
        , config.weight_fn)



class Trainer:
    def __init__(self, config):
        self.config = config

    def train(self, model, crit, optimizer, train_loader, valid_loader):
        train_engine = MyEngine(MyEngine.train, model, crit, optimizer, self.config)
        validate_engine = MyEngine(MyEngine.validate, model, crit, optimizer, self.config)

        MyEngine.attach(train_engine, validate_engine, verbose=self.config.verbose)

        def run_validation(v_engine, v_loader):
            v_engine.run(v_loader, max_epochs=1)

        train_engine.add_event_handler(Events.EPOCH_COMPLETED, run_validation, validate_engine, valid_loader)
        validate_engine.add_event_handler(Events.EPOCH_COMPLETED, MyEngine.check_best)
        validate_engine.add_event_handler(Events.EPOCH_COMPLETED, MyEngine.save_model, self.config)

        train_engine.run(train_loader, max_epochs=self.config.n_epochs)

        model.load_state_dict(validate_engine.best_model)

        return model
