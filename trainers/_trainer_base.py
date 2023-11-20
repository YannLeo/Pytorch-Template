# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:18:02
# @File       : _trainer_base.py
# @Note       : The base class of all trainers, and some tools for training

from typing import NamedTuple, Iterator, Callable, Any
from abc import ABC, abstractmethod
import itertools
import logging
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
import rich.progress

import datasets
import models


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
    metric: int | float = 0
    color: str = "k"


class _TrainerBase(ABC):
    LINE_UP = "\033[1A"  # ANSI escape code: move cursor up one line

    def __init__(self, info: dict[str, Any], path: Path = Path(), device=torch.device("cuda")):
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
        self.info: dict = info  # dict of configs from toml file
        self.resume: str | None = info.get("resume")  # path to checkpoint
        self.device: torch.device = device
        self.max_epoch: int = info["epochs"]
        self.num_classes: int = info["num_classes"]
        self.log_path: Path = path / "log" / "log.txt"
        self.model_path: Path = path / "model"
        self.confusion_path: Path = path / "confusion_matrix"
        self.save_period: int = info["save_period"]
        self._min_test_loss: float = np.inf
        self._min_valid_pretrain_loss: float = np.inf

        self.model: torch.nn.Module  # must be defined in `self._prepare_models()`

        # temp variables for confusion matrix
        self._y_true: list = []
        self._y_pred: list = []

        # 1. Dataloaders
        self._prepare_dataloaders()                    
        self._adapt_epoch_to_step()  # adapt epoch_size to step_size               
        # 2. Defination and initialization of the models
        self._prepare_models()
        # 3. Optimizers and schedulers of the models
        self._prepare_opt()
        # loggers
        self._get_logger()  # txt logger
        self.metrics_writer: SummaryWriter = SummaryWriter(path / "log")

    def _prepare_dataloaders(self) -> None:
        """
        1. Prepare dataloaders for training (single) and testing (multiple).
        """
        # Load training dataloader, if exists
        try:
            train_loader_config = self.info["dataloader_train"]  # single dataloader for training
            # Load training dataloader
            self.dataset_train = self._get_object(datasets, train_loader_config["dataset"]["name"], train_loader_config["dataset"]["args"])
            self.dataloader_train = DataLoader(dataset=self.dataset_train, **train_loader_config["args"])
            self.train_steps = len(self.dataloader_train) 
        except KeyError:
            print("Skipping loading training dataloader because of lacking key in toml: `dataloader_train`.")

        # Load testing dataloaders as a dict, sorted by name
        test_loaders = list(filter(lambda x: "dataloader_test" in x, self.info))  # multiple dataloaders for testing
        dataset_test_dict = {
            test_loader_name: self._get_object(
                datasets, self.info[test_loader_name]["dataset"]["name"], self.info[test_loader_name]["dataset"]["args"]
            )
            for test_loader_name in test_loaders
        }
        self.dataloader_test_dict = {
            test_loader_name: DataLoader(dataset=dataset_test_dict[test_loader_name], **self.info[test_loader_name]["args"])
            for test_loader_name in test_loaders
        }

    def _prepare_models(self) -> None:
        """
        2. Prepare models for training. The name `self.model` is reserved for some functions in the base class
        """
        self.model: torch.nn.Module = self._get_object(models, self.info["model"]["name"], self.info["model"]["args"])
        self._resuming_model(self.model)  # Prepare for resuming models, will not resume optimizers and schedulers!!
        self.model = self.model.to(self.device)

    def _prepare_opt(self) -> None:
        """
        3. Prepare optimizers and corresponding learning rate schedulers.
        """
        self.opt = torch.optim.AdamW(params=self.model.parameters(), lr=self.info["lr_scheduler"]["init_lr"])
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = self._get_object(
            torch.optim.lr_scheduler, self.info["lr_scheduler"]["name"], {"optimizer": self.opt, **self.info["lr_scheduler"]["args"]}
        )

    def reset_grad(self) -> None:
        """
        Reset gradients of all trainable parameters.
        """
        self.opt.zero_grad(set_to_none=True)

    def _resuming_model(self, model: torch.nn.Module) -> None:
        """Note: only for variable `self.model`"""
        self.epoch = 1
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict)
            # self.epoch = checkpoint["epoch"] + 1

    def train(self) -> None:  # sourcery skip: low-code-quality
        """
        Call train_epoch() and test_epoch() for each epoch and log the results.
        The best model is defined by the test loss of the FIRST dataset_test by default.
        Change the behavior self._epoch_end().
        """
        test_loaders: list[str] = sorted(self.dataloader_test_dict.keys())
        for epoch in range(self.epoch, self.max_epoch + 1):
            time_flag = time.time()

            """1. Training epoch"""
            metrics_train = self.train_epoch(epoch)
            _str = f"Epoch: {epoch:<4d}| {self.metrics_wrapper(metrics_train, with_color=True)}"
            print(_str + ("Testing..." if test_loaders else ""), end="\n" if test_loaders else "")

            """2. Testing epoch(s)"""
            list_metrics_test: list[dict[str, metrics]] = []
            metrics_test: dict[str, metrics] = {}
            for idx, test_loader in enumerate(test_loaders):
                # Why type(self)? Because `func_to_plot_confusion` in `plot_confusion` will add `self` as the first argument
                # when use `plot_confusion(self.test_epoch)`. But using `@plot_confusion()` before `self.test_epoch(self)` will not.
                test_epoch = plot_confusion(name=test_loader.split("_")[-1], interval=5)(type(self).test_epoch)
                metrics_test = test_epoch(self, epoch, self.dataloader_test_dict[test_loader])
                list_metrics_test.append(metrics_test)
                _str += f"{self.metrics_wrapper(metrics_test, with_color=True)}"
                print(
                    self.LINE_UP + _str + ("Testing..." if idx != len(test_loaders) - 1 else ""),
                    end="\n" if idx != len(test_loaders) - 1 else "",
                )

                """3. Logging results"""
                # need to be specified by yourself
                self.metrics_writer.add_scalar(f"test_acc{idx}", metrics_test["test_acc"].metric, global_step=epoch)

            # The end of each epoch
            print(f"time:{int(time.time() - time_flag):3d}s", end="")
            self._epoch_end(epoch, metrics_train, list_metrics_test)  # Must be called at the end of each epoch

        # The end of the whole training
        self._train_end()  # Must be called at the end of the training

    def _epoch_end(
        self, epoch: int, metrics_train: dict[str, metrics], list_metrics_test: list[dict[str, metrics]], *args, **kwargs
    ) -> None:
        self.epoch = epoch
        # Need to be specified by yourself:
        # which dataloader_test (first, i.e., 0 by default) to use and which metric (tset_loss by default) to use.
        if list_metrics_test:
            best = self._save_model_by_test_loss(epoch, list_metrics_test[0]["test_loss"].metric)
        else:
            best = self._save_model_by_test_loss(epoch, metrics_train["train_loss"].metric)
        # Logging to log.txt
        self.logger.info(
            f"Epoch: {epoch:<4d}| {self.metrics_wrapper(metrics_train)}"
            + "".join([f"{self.metrics_wrapper(metrics)}" for metrics in list_metrics_test])
            + f'{"saving best model..." if best else ""}',
        )

    def _train_end(self) -> None:
        """If this function is overrided, please call super()._train_end() at the end of the function."""
        self.metrics_writer.close()

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, metrics]:
        pass

    @abstractmethod
    def test_epoch(self, epoch: int, dataloader_test: DataLoader) -> dict[str, metrics]:
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

    @staticmethod
    def _plot_confusion_matrix_impl(
        photo_path: str | Path,
        labels: np.ndarray,
        predicts: np.ndarray,
        classes: list,
        normalize: bool = True,
        title: str = "Confusion Matrix",
        cmap=plt.cm.Oranges,  # type: ignore
    ) -> None:
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
    def _get_object(module, s: str, parameter: dict) -> Any:
        return getattr(module, s)(**parameter)

    def _get_logger(self) -> None:
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

    def _save_model_by_test_loss(self, epoch: int, test_loss: float) -> bool:
        flag = False
        if test_loss < self._min_test_loss:  # meet the condition to save the model
            flag = True
            self._min_test_loss = test_loss
            if epoch % self.save_period == 0:
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

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
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

    def _adapt_epoch_to_step(self) -> None:
        if self.train_steps is None:
            self.train_steps = len(self.dataloader_train)
        for lr_scheduler in filter(lambda x: "lr_scheduler" in x, self.info):
            params = self.info[lr_scheduler]["args"]
            if params.get("epoch_size"):  # get epoch_size rather than step_size
                params["step_size"] = int(params["epoch_size"] * self.train_steps)
                params.pop("epoch_size")

    def progress(self, dataloader: DataLoader|zip, epoch: int, test: bool=False, total: int|None=None) -> Iterator:
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
                total = len(dataloader) # type: ignore
            except TypeError:
                __key_1st_testloader = sorted(self.dataloader_test_dict.keys())[0]
                total = len(self.dataloader_test_dict[__key_1st_testloader]) if test else self.train_steps

        with _progress:
            description = "Testing" if test else f"Epoch {epoch}/{self.max_epoch}"
            yield from _progress.track(dataloader, total=total, description=description, update_period=0.1)
            _progress.update(rich.progress.TaskID(0), description=f"[green]Epoch {epoch:<2d}")


# A decorator to plot confusion matrix easily
def plot_confusion(name: str = "test", interval: int = 1) -> Callable[..., Callable[..., dict[str, metrics]]]:
    def decorator(func_to_plot_confusion: Callable[..., dict[str, metrics]]):
        # wrapper to the actual function, e.g. self.test_epoch(self, epoch, *args, **kwargs)
        def wrapper(self: _TrainerBase, epoch, *args, **kwargs):
            # 1. before the func: empty the public list
            self._y_pred, self._y_true = [], []
            _y_pred_np: np.ndarray = np.array([])
            _y_true_np: np.ndarray = np.array([])
            # 2. call the func
            metrics = func_to_plot_confusion(self, epoch, *args, **kwargs)
            # 3. after the func: check and plot the confusion matrix
            if epoch % interval == 0 and (len(self._y_pred) + len(self._y_true)) > 0:
                if isinstance(self._y_pred, list) or isinstance(self._y_true, list):
                    _y_pred_np = np.concatenate(self._y_pred, axis=0)
                    _y_true_np = np.concatenate(self._y_true, axis=0)
                self._plot_confusion_matrix_impl(
                    photo_path=self.confusion_path / f"{name}-{str(epoch).zfill(len(str(self.max_epoch)))}.png",
                    labels=_y_true_np,
                    predicts=_y_pred_np,
                    classes=list(range(self.num_classes)),
                )
            return metrics

        return wrapper

    return decorator
