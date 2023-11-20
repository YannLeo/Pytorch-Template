# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:20:01
# @File       : basic_trainer.py
# @Note       : A basic trainer for training a feed forward neural network

from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from ._trainer_base import _TrainerBase, plot_confusion, metrics


class BasicTrainer(_TrainerBase):
    def __init__(self, info: dict[str, Any], path=Path(), device=torch.device("cuda")) -> None:
        """(1)Dataloaders, (2)models, (3)optimizers along with schedulers and (4)loggers are prepared in super().__init__() in sequence."""
        super().__init__(info, path, device)
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=info["label_smoothing"])

    @plot_confusion(name="train", interval=6)  # "train.png"
    def train_epoch(self, epoch: int) -> dict[str, metrics]:  # sourcery skip: low-code-quality
        """
        The main training process, which will be called in self.train()
        self._y_true & self._y_pred is for plotting confusion matrix
        """
        # helper variables
        num_samples, num_correct = 0, 0
        train_loss = 0.0
        data: torch.Tensor
        targets: torch.Tensor

        self.model.train()  # don't forget
        for batch, (data, targets) in enumerate(self.progress(self.dataloader_train, epoch=epoch)):
            self._y_true.append(targets.numpy())  # for plotting confusion matrix
            data, targets = data.to(self.device), targets.to(self.device)

            # 1. Forwarding
            output: torch.Tensor = self.model(data)
            # 2. Computing loss
            loss: torch.Tensor = self.loss_func(output, targets)
            # 3. Backwarding: compute gradients and update parameters
            self.reset_grad()
            loss.backward()
            self.opt.step()
            # 4. Updating learning rate by step; move it to self.train() if you want to update lr by epoch
            self.metrics_writer.add_scalar("lr", self.opt.param_groups[0]["lr"], global_step=epoch * self.train_steps + batch)
            self.lr_scheduler.step()
            # 5. Computing metrics
            num_samples += data.shape[0]
            train_loss += loss.item()
            predicts = output.argmax(dim=1)
            num_correct += torch.sum(predicts == targets).item()

            self._y_pred.append(predicts.cpu().numpy())  # for plotting confusion matrix

        return {
            "train_loss": metrics(train_loss / len(self.dataloader_train)),
            "train_acc": metrics(num_correct / num_samples, "blue"),
        }

    @torch.inference_mode()  # disable autograd
    def test_epoch(self, epoch: int, dataloader_test: DataLoader) -> dict[str, metrics]:
        """
        The main testing process, which will be called in self.train()
        self._y_true & self._y_pred is for plotting confusion matrix
        """
        num_correct, num_samples = 0, 0
        test_loss = 0.0
        data: torch.Tensor
        targets: torch.Tensor

        self.model.eval()  # don't forget!
        for data, targets in self.progress(dataloader_test, epoch=epoch, test=True):
            self._y_true.append(targets.numpy())  # for plotting confusion matrix
            data, targets = data.to(self.device), targets.to(self.device)
            # Forwarding
            output: torch.Tensor = self.model(data)
            # Computing metrics
            test_loss += self.loss_func(output, targets).item()
            num_samples += data.shape[0]
            predicts = output.argmax(dim=1)
            num_correct += torch.sum(predicts == targets).item()

            self._y_pred.append(predicts.cpu().numpy())  # for plotting confusion matrix

        return {
            "test_loss": metrics(test_loss / len(dataloader_test)),
            "test_acc": metrics(num_correct / num_samples, "red"),
        }
