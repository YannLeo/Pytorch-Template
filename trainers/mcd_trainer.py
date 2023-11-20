# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 23/01/11 ~ 22:07:25
# @File       : mcd_trainer.py
# @Note       : A simple implementation of MCD (Maximum Classifier Discrepancy for
#               Unsupervised Domain Adaptation) in PyTorch


from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from ._trainer_base import _TrainerBase, plot_confusion, metrics
import models
import datasets


class MCDTrainer(_TrainerBase):
    """
    A simple implementation of MCD (https://arxiv.org/abs/1712.02560). The most essential
    part of the code are the functions train_epoch() and test_epoch().

    Ref: https://github.com/mil-tokyo/MCD_DA
    """

    def __init__(self, info: dict[str, Any], path=Path(), device=torch.device("cuda")) -> None:
        """(1)Dataloaders, (2)models, (3)optimizers along with schedulers and (4)loggers are prepared in super().__init__() in sequence."""
        super().__init__(info, path, device)
        self.loss_func = nn.CrossEntropyLoss()
        self.discrepancy_steps: int = info["discrepancy_steps"]
        self.discrepancy_weight: float = info["discrepancy_weight"]

    def _prepare_dataloaders(self) -> None:
        """
        Prepare the dataloaders for the source and target domains.
        """
        super()._prepare_dataloaders()
        # 1. datasets of source domain
        self.dataset_source = self._get_object(
            datasets, self.info["dataloader_source"]["dataset"]["name"], self.info["dataloader_source"]["dataset"]["args"]
        )
        self.dataloader_source = DataLoader(dataset=self.dataset_source, **self.info["dataloader_source"]["args"])
        self.dataset_val = self._get_object(
            datasets, self.info["dataloader_val"]["dataset"]["name"], self.info["dataloader_val"]["dataset"]["args"]
        )
        self.dataloader_val = DataLoader(dataset=self.dataset_val, **self.info["dataloader_val"]["args"])
        # 2.1 dataset of target domain
        self.dataset_target = self._get_object(
            datasets, self.info["dataloader_target"]["dataset"]["name"], self.info["dataloader_target"]["dataset"]["args"]
        )
        self.dataloader_target = DataLoader(dataset=self.dataset_target, **self.info["dataloader_target"]["args"])
        # 2.2 datasets of test set of target domain (done by super()._prepare_dataloaders())

        # Convert epoch_size to step_size
        self.train_steps = min(len(self.dataloader_source), len(self.dataloader_target))

    def _prepare_models(self) -> None:
        """
        Prepare the models.
        """
        # the name `self.model` is reserved for some functions in the base class
        super()._prepare_models()
        self.C1 = models.Classifier(
            input_dim=self.info["model"]["args"]["num_classes"], num_class=self.num_classes, intermediate_dim=128, layers=3
        )
        self.C2 = models.Classifier(
            input_dim=self.info["model"]["args"]["num_classes"], num_class=self.num_classes, intermediate_dim=128, layers=2
        )
        self.C1 = self.C1.to(self.device)
        self.C2 = self.C2.to(self.device)

    def _prepare_opt(self) -> None:
        """
        Prepare the optimizers and corresponding learning rate schedulers.
        """
        super()._prepare_opt()
        self.opt_C1 = torch.optim.Adam(params=self.C1.parameters(), lr=self.info["lr_scheduler_C"]["init_lr"])
        self.opt_C2 = torch.optim.Adam(params=self.C2.parameters(), lr=self.info["lr_scheduler_C"]["init_lr"])
        self.lr_scheduler_C1: torch.optim.lr_scheduler.LRScheduler = self._get_object(
            torch.optim.lr_scheduler, self.info["lr_scheduler_C"]["name"], {"optimizer": self.opt_C1, **self.info["lr_scheduler_C"]["args"]}
        )
        self.lr_scheduler_C2: torch.optim.lr_scheduler.LRScheduler = self._get_object(
            torch.optim.lr_scheduler, self.info["lr_scheduler_C"]["name"], {"optimizer": self.opt_C2, **self.info["lr_scheduler_C"]["args"]}
        )

    def reset_grad(self) -> None:
        """
        Reset gradients of all trainable parameters.
        """
        super().reset_grad()
        self.opt_C1.zero_grad(set_to_none=True)
        self.opt_C2.zero_grad(set_to_none=True)

    def train_epoch(self, epoch):  # sourcery skip: low-code-quality
        """
        Main training process
        """
        # Helper variables
        train_loss_discrepancy = 0
        train_loss_C1, train_loss_C2 = 0, 0
        num_samples = 0  # source domain only
        num_correct_C1_src, num_correct_C2_src = 0, 0  # on source domain
        num_correct_tgt = 0  # ensemble and comprehensive result of C1 and C2

        self.model.train()
        self.C1.train()
        self.C2.train()
        loop = enumerate(self.progress(zip(self.dataloader_source, self.dataloader_target), epoch=epoch, total=self.train_steps))
        for batch, ((data_s, label_s), (data_t, label_t)) in loop:  # label_t is merely for metrics
            data_s, data_t = data_s.to(self.device), data_t.to(self.device)
            label_s, label_t = label_s.to(self.device), label_t.to(self.device)
            num_samples += data_s.shape[0]

            """step 1. Training on source domain only"""
            feature_s: torch.Tensor = self.model(data_s)
            output_s1: torch.Tensor = self.C1(feature_s)
            output_s2: torch.Tensor = self.C2(feature_s)
            loss_s1: torch.Tensor = self.loss_func(output_s1, label_s)
            loss_s2: torch.Tensor = self.loss_func(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            self.reset_grad()
            loss_s.backward()
            self.opt.step()
            self.opt_C1.step()
            self.opt_C2.step()

            """step 2. Maximizing discrepancy"""
            feature_s = self.model(data_s)
            output_s1 = self.C1(feature_s)
            output_s2 = self.C2(feature_s)
            feature_t = self.model(data_t)
            output_t1 = self.C1(feature_t)
            output_t2 = self.C2(feature_t)
            loss_s1 = self.loss_func(output_s1, label_s)
            loss_s2 = self.loss_func(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis * self.discrepancy_weight
            self.reset_grad()
            loss.backward()
            self.opt_C1.step()
            self.opt_C2.step()
            # Computing metrics
            train_loss_C1 += loss_s1.item()
            train_loss_C2 += loss_s2.item()
            num_correct_C1_src += (output_s1.argmax(dim=1) == label_s).sum().item()
            num_correct_C2_src += (output_s2.argmax(dim=1) == label_s).sum().item()

            """step 3. Minimizing discrepancy"""
            for _ in range(self.discrepancy_steps):
                feature_t = self.model(data_t)
                output_t1 = self.C1(feature_t)
                output_t2 = self.C2(feature_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                self.reset_grad()
                loss_dis.backward()
                self.opt.step()
            # Computing metrics
            num_correct_tgt += ((output_t1 + output_t2).argmax(dim=1) == label_t).sum().item()
            train_loss_discrepancy += loss_dis.item()

            # Updating learning rate by step
            self.metrics_writer.add_scalar("lr", self.opt.param_groups[0]["lr"], global_step=epoch * self.train_steps + batch)
            self.lr_scheduler.step()
            self.lr_scheduler_C1.step()
            self.lr_scheduler_C2.step()

        return {
            "train_loss_C1": metrics(train_loss_C1 / self.train_steps),
            "train_loss_C2": metrics(train_loss_C2 / self.train_steps),
            "train_loss_dis": metrics(train_loss_discrepancy / self.train_steps),
            "acc_C1_s": metrics(num_correct_C1_src / num_samples),
            "acc_C2_s": metrics(num_correct_C2_src / num_samples),
            "acc_tgt": metrics(num_correct_tgt / num_samples, "green"),
        }

    @torch.inference_mode()  # disable autograd
    def test_epoch(self, epoch: int, dataloader_test: DataLoader) -> dict[str, metrics]:
        """
        The main testing process, which will be called in self.train()
        self._y_true & self._y_pred is for plotting confusion matrix
        """
        # Helper variables
        num_correct, num_correct_C1, num_correct_C2, num_samples = 0, 0, 0, 0
        test_loss = 0.0
        data: torch.Tensor
        targets: torch.Tensor

        self.model.eval()
        self.C1.eval()
        self.C2.eval()
        for data, targets in self.progress(dataloader_test, epoch=epoch, test=True):
            self._y_true.append(targets.numpy())

            data, targets = data.to(self.device), targets.to(self.device)

            # Forwarding
            feature: torch.Tensor = self.model(data)
            output1: torch.Tensor = self.C1(feature)
            output2: torch.Tensor = self.C2(feature)
            # Computing metrics
            num_samples += data.shape[0]
            test_loss += (self.loss_func(output1, targets) + self.loss_func(output2, targets)).item() / 2
            predicts = torch.argmax(output1 + output2, dim=1)  # ensemble
            num_correct += torch.sum(predicts == targets).item()
            num_correct_C1 += torch.sum(torch.argmax(output1, dim=1) == targets).item()
            num_correct_C2 += torch.sum(torch.argmax(output2, dim=1) == targets).item()

            self._y_pred.append(predicts.cpu().numpy())

        return {
            "test_loss": metrics(test_loss / len(dataloader_test)),
            "test_acc": metrics(num_correct / num_samples, "blue"),
            "test_acc_C1": metrics(num_correct_C1 / num_samples),
            "test_acc_C2": metrics(num_correct_C2 / num_samples),
        }

    @staticmethod
    def discrepancy(out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(torch.softmax(out1, dim=1) - torch.softmax(out2, dim=1)))
