# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:18:02
# @File       : _trainer_base.py
# @Note       : The base class of all trainers, and some tools for training

from typing import NamedTuple
from abc import ABC, abstractmethod
import itertools
import logging
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import time
import rich.progress


# Used to print the log in different colors: r, g, b, w, c, m, y, k
_color_map = {
    "black": ("\033[30m", "\033[0m"),  # 黑色字
    "k": ("\033[30m", "\033[0m"),  # 黑色字
    "red": ("\033[31m", "\033[0m"),  # 红色字
    "r": ("\033[31m", "\033[0m"),  # 红色字
    "green": ("\033[32m", "\033[0m"),  # 绿色字
    "g": ("\033[32m", "\033[0m"),  # 绿色字
    "yellow": ("\033[33m", "\033[0m"),  # 黄色字
    "y": ("\033[33m", "\033[0m"),  # 黄色字
    "blue": ("\033[34m", "\033[0m"),  # 蓝色字
    "b": ("\033[34m", "\033[0m"),  # 蓝色字
    "magenta": ("\033[35m", "\033[0m"),  # 紫色字
    "m": ("\033[35m", "\033[0m"),  # 紫色字
    "cyan": ("\033[36m", "\033[0m"),  # 青色字
    "c": ("\033[36m", "\033[0m"),  # 青色字
    "white": ("\033[37m", "\033[0m"),  # 白色字
    "w": ("\033[37m", "\033[0m"),  # 白色字
    "white_on_k": ("\033[40;37m", "\033[0m"),  # 黑底白字
    "white_on_r": ("\033[41;37m", "\033[0m"),  # 红底白字
    "white_on_g": ("\033[42;37m", "\033[0m"),  # 绿底白字
    "white_on_y": ("\033[43;37m", "\033[0m"),  # 黄底白字
    "white_on_b": ("\033[44;37m", "\033[0m"),  # 蓝底白字
    "white_on_m": ("\033[45;37m", "\033[0m"),  # 紫底白字
    "white_on_c": ("\033[46;37m", "\033[0m"),  # 天蓝白字
    "black_on_w": ("\033[47;30m", "\033[0m"),  # 白底黑字
}


class metrics(NamedTuple):
    metric: int | float
    color: str = "k"


class _Trainer_Base(ABC):
    def __init__(self, info: dict, path: Path = Path(), device=torch.device("cuda")):
        """
        Initialize the model, optimizer, scheduler, dataloader, and logger.

        Args:
          info (dict): dict of configs from toml file
          resume: path to checkpoint
          path: the path to the folder where the model will be saved.
          device: the device to run the model on.

        ---
        Tips: This function is highly dependent on the config file, where the data, models, optimizers and
              schedulers are defined.
        """
        # Basic constants
        self.info = info  # dict of configs from toml file
        self.resume = info.get("resume")  # path to checkpoint
        self.device = device
        self.max_epoch = info["epochs"]
        self.num_classes = info["num_classes"]
        self.log_path = path / "log" / "log.txt"
        self.model_path = path / "model"
        self.confusion_path = path / "confusion_matrix"
        self.save_period = info["save_period"]
        self._min_valid_loss = np.inf
        self._min_valid_pretrain_loss = np.inf

        self.model: torch.nn.Module | None = None  # must be defined in `self.__prepare_models()`

        # temp variables for confusion matrix
        self._y_true: list = []
        self._y_pred: list = []

        # 1. Dataloaders
        self._prepare_dataloaders()
        # 2. Defination and initialization of the models
        self._prepare_models()
        # 3. Optimizers and schedulers of the models
        self._prepare_opt()
        # loggers
        self._get_logger()  # txt logger
        self.metrics_writer = SummaryWriter(path / "log")

    def _prepare_dataloaders(self):
        ...

    @abstractmethod
    def _prepare_models(self):
        pass

    def _resuming_model(self, model: torch.nn.Module):
        """Note: only for variable `self.model`"""
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict)
            self.epoch = checkpoint["epoch"] + 1
        else:
            self.epoch = 0

    @abstractmethod
    def _prepare_opt(self):
        pass

    @abstractmethod
    def _reset_grad(self):
        pass

    @staticmethod
    def metrics_wrapper(metrics: dict[str, metrics], with_color: bool = False) -> str:
        """
        It takes a dictionary of metrics and returns a string of the metrics in a nice format

        Args:
          metrics (dict): dict[str, metrics]
          with_color: If True, the metrics will be printed with color codes (see -> `_color_map`). Defaults to False

        Returns:
          A string of the metrics
        """
        return (
            "".join(
                f"{_color_map[metric.color][0]}{key}: {metric.metric:.4f}{_color_map[metric.color][1]} | "
                for key, metric in metrics.items()
            )
            if with_color
            else "".join(f"{key}: {metric.metric:.4f} | " for key, metric in metrics.items())
        )

    def train(self):  # sourcery skip: low-code-quality
        """
        Call train_epoch() and test_epoch() for each epoch and log the results.
        """
        for epoch in range(self.epoch, self.max_epoch):
            time_begin = time.time()

            """1. Training epoch"""
            metrics_train = self.train_epoch(epoch)
            print(f"Epoch: {epoch:<4d}| {self.metrics_wrapper(metrics_train, with_color=True)}Testing...")

            # time.sleep(1)
            """2. Testing epoch"""
            metrics_test = self.test_epoch(epoch)
            time_end = time.time()
            print("\x1b\x4d" * 2)  # move cursor up
            print(f"Epoch: {epoch:<4d}| {self.metrics_wrapper(metrics_train, with_color=True)}", end="")
            print(f"{self.metrics_wrapper(metrics_test, with_color=True)}time:{int(time_end - time_begin):3d}s", end="", flush=True)

            """3. Logging results"""
            best = self._save_model_by_test_loss(epoch, metrics_test["test_loss"])  # need to be specified by yourself
            self.metrics_writer.add_scalar("test_acc", metrics_test["test_acc"][0], global_step=epoch)  # need to be specified by yourself
            # log to log.txt
            self.logger.info(
                f"Epoch: {epoch:<4d}| "
                f"{self.metrics_wrapper(metrics_train)}{self.metrics_wrapper(metrics_test)}"
                f'{"saving best model..." if best else ""}'
            )
            self._epoch_end(epoch)  # Can be called at the end of each epoch
            self.epoch += 1

        self._train_end()  # Must be called at the end of the training

    def _epoch_end(self, epoch):
        pass

    def _train_end(self):
        """If this function is overrided, please call super()._train_end() at the end of the function."""
        self.metrics_writer.close()

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, metrics]:
        pass

    @abstractmethod
    def test_epoch(self, epoch) -> dict[str, metrics]:
        pass

    @staticmethod
    def _plot_confusion_matrix_impl(photo_path, labels, predicts, classes, normalize=True, title="Confusion Matrix", cmap=plt.cm.Oranges):
        FONT_SIZE = 13
        cm = confusion_matrix(labels, predicts, labels=list(range(len(classes))))
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={"float": "{: 0.2f}".format})
        plt.figure(figsize=(11, 9))
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title, fontsize=FONT_SIZE)
        plt.colorbar(extend=None)
        plt.clim(0, 1)
        plt.xticks(np.arange(len(classes)), classes, rotation=25, fontsize=FONT_SIZE - 2)
        plt.yticks(np.arange(len(classes)), classes, fontsize=FONT_SIZE - 2)
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                fontsize=FONT_SIZE,
                color="white" if cm[i, j] > thresh else "black",
            )
        plt.tight_layout()
        plt.ylabel(r"$True\; labels$", fontsize=FONT_SIZE)
        plt.xlabel(r"$Predicted\; labels$", fontsize=FONT_SIZE)
        plt.savefig(photo_path, format="png", bbox_inches="tight", dpi=100)

    @staticmethod
    def _get_object(module, s: str, parameter: dict):
        return getattr(module, s)(**parameter)

    def _get_logger(self):
        self.logger = logging.getLogger("train")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info(
            f'{time.strftime("%Y-%m-%d %p %A", time.localtime())} - ' f"model: {type(self.model).__name__}"
        )  # Only log name of variable `self.model`

    def _save_model_by_test_loss(self, epoch, valid_loss) -> bool:
        flag = 0
        if valid_loss < self._min_valid_loss:
            flag = 1
            self._min_valid_loss = valid_loss
            if epoch % self.save_period == (__period := self.save_period - 1):
                print(" | saving best model and checkpoint...")
                self._save_checkpoint(epoch, True)
                self._save_checkpoint(epoch, False)
            else:
                print(" | saving best model...")
                self._save_checkpoint(epoch, True)
        elif epoch % self.save_period == 0:
            print(" | saving checkpoint...")
            self._save_checkpoint(epoch, False)
        else:
            print()
        return flag

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),  # Only save parameters of variable `self.model`
            "loss_best": self._min_valid_pretrain_loss,
        }
        if save_best:
            best_path = str(self.model_path / ("model_best.pth"))
            torch.save(state, best_path)
        else:
            path = str(self.model_path / f"checkpoint-epoch{epoch}.pth")
            torch.save(state, path)

    @staticmethod
    def _adapt_epoch_to_step(params: dict, train_steps: int = None):
        if params.get("epoch_size", False):  # get epoch_size rather than step_size
            params["step_size"] = int(params["epoch_size"] * train_steps)
            params.pop("epoch_size")

    def progress(self, dataloader, epoch, test=False, total=None):
        _progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.percentage]{task.description}"),
            rich.progress.SpinnerColumn("dots" if test else "moon", "progress.percentage", finished_text="[green]✔"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn("[progress.download]{task.percentage:>5.1f}%"),
            rich.progress.MofNCompleteColumn(),
            # TaskProgressColumn("[progress.percentage]{task.completed:>3d}/{task.total:<3d}"),
            "•[progress.remaining] ⏳",
            rich.progress.TimeRemainingColumn(),
            "•[progress.elapsed] ⏰",
            rich.progress.TimeElapsedColumn(),
            transient=True,
        )
        if total is None:
            try:
                total = len(dataloader)
            except TypeError:
                total = self.num_batches_test if test else self.num_batches_train

        with _progress:
            description = "Testing" if test else f"Epoch {epoch+1}/{self.max_epoch}"
            yield from _progress.track(dataloader, total=total, description=description, update_period=0.1)
            _progress.update(0, description=f"[green]Epoch {epoch+1:<2d}")


# a decorator to plot confusion matrix easily
def plot_confusion(name="test", interval=1):
    def decorator(func_to_plot_confusion):
        # wrapper to the actual function, e.g. self.test_epoch(self, epoch, *args, **kwargs)
        def wrapper(self, epoch, *args, **kwargs):
            # 1. before the func: empty the public list
            self._y_pred, self._y_true = [], []
            # 2. call the func
            metrics = func_to_plot_confusion(self, epoch, *args, **kwargs)
            # 3. after the func: check and plot the confusion matrix
            if epoch % interval == 0 and (len(self._y_pred) + len(self._y_true)) > 0:
                if isinstance(self._y_pred, list) or isinstance(self._y_true, list):
                    self._y_pred = np.concatenate(self._y_pred, axis=0)
                    self._y_true = np.concatenate(self._y_true, axis=0)
                self._plot_confusion_matrix_impl(
                    photo_path=self.confusion_path / f"{name}-{str(epoch).zfill(len(str(self.max_epoch)))}.png",
                    labels=self._y_true,
                    predicts=self._y_pred,
                    classes=list(range(self.num_classes)),
                )
            return metrics

        return wrapper

    return decorator
