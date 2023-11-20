# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 23/01/07 ~ 17:02:47
# @File       : dann_trainer.py
# @Note       : A simple implementation of DANN (Domain-Adversarial Training of Neural Networks)
#               in PyTorch. This file overrides many functions to customize the whole process.


from typing import Annotated, Union, Any
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from ._trainer_base import _TrainerBase, plot_confusion, metrics
import models
import datasets


class _GradientReverseFunction(torch.autograd.Function):
    """
    Helper class for gradient reversal layer (_GradientReverseLayer)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff=1.0) -> torch.Tensor:
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.coeff, None


class _GradientReverseLayer(nn.Module):
    """
    Ref: https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/grl.py

    Args:
      coeff: The coefficient of the reversed gradient, and must be a positive number.
    """

    def __init__(self, coeff: Annotated[float, "it is positive"] = 1.0):
        super().__init__()
        self.coeff = coeff

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, None]:
        return _GradientReverseFunction.apply(x, self.coeff)


class DANNTrainer(_TrainerBase):
    """
    A simple implementation of DANN (https://arxiv.org/abs/1505.07818) in PyTorch. The
    most essential part of the code are the functions train_epoch() and test_epoch().
    """

    def __init__(self, info: dict[str, Any], path=Path(), device=torch.device("cuda")) -> None:
        """(1)Dataloaders, (2)models, (3)optimizers along with schedulers and (4)loggers are prepared in super().__init__() in sequence."""
        super().__init__(info, path, device)
        self.loss_func = nn.CrossEntropyLoss()
        self.grl = _GradientReverseLayer(coeff=info["GRL_coeff"])

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
        self.classifier_content = models.Classifier(
            input_dim=self.info["model"]["args"]["num_classes"], num_class=self.num_classes, intermediate_dim=128, layers=2
        )
        self.classifier_domain = models.Classifier(
            input_dim=self.info["model"]["args"]["num_classes"], num_class=self.num_classes, intermediate_dim=128, layers=2
        )
        self.classifier_content = self.classifier_content.to(self.device)
        self.classifier_domain = self.classifier_domain.to(self.device)

    def _prepare_opt(self) -> None:
        """
        Prepare the optimizers and corresponding learning rate schedulers.
        """
        super()._prepare_opt()
        self.opt_content = torch.optim.Adam(params=self.classifier_content.parameters(), lr=self.info["lr_scheduler"]["init_lr"] / 2)
        self.opt_domain = torch.optim.Adam(params=self.classifier_domain.parameters(), lr=self.info["lr_scheduler"]["init_lr"] / 2)
        self.lr_scheduler_content: torch.optim.lr_scheduler.LRScheduler = self._get_object(
            torch.optim.lr_scheduler,
            self.info["lr_scheduler"]["name"],
            {"optimizer": self.opt_content, **self.info["lr_scheduler"]["args"]},
        )
        self.lr_scheduler_domain: torch.optim.lr_scheduler.LRScheduler = self._get_object(
            torch.optim.lr_scheduler, self.info["lr_scheduler"]["name"], {"optimizer": self.opt_domain, **self.info["lr_scheduler"]["args"]}
        )

    def reset_grad(self) -> None:
        """
        Reset gradients of all trainable parameters.
        """
        super().reset_grad()
        self.opt_content.zero_grad(set_to_none=True)
        self.opt_domain.zero_grad(set_to_none=True)

    @plot_confusion(name="train", interval=999)  # "train.png
    def train_epoch(self, epoch: int) -> dict[str, metrics]:  # sourcery skip: low-code-quality
        """
        Main training process
        """
        # Helper variables
        train_loss_content_src = 0
        num_samples = 0  # source domain only
        num_correct_content, num_correct_domain = 0, 0

        self.model.train()
        self.classifier_content.train()
        self.classifier_domain.train()

        loop = enumerate(self.progress(zip(self.dataloader_source, self.dataloader_target), epoch=epoch, total=self.train_steps))
        for batch, ((data_s, label_s), (data_t, _)) in loop:  # we do not use label_t here
            data_s, data_t = data_s.to(self.device), data_t.to(self.device)
            label_s: torch.Tensor = label_s.to(self.device)
            # Domain labels
            label_ones = torch.ones(len(data_s), dtype=torch.long, device=self.device)  # src domain
            label_zeros = torch.zeros(len(data_t), dtype=torch.long, device=self.device)  # tgt domain

            feature_s: torch.Tensor = self.model(data_s)
            feature_t: torch.Tensor = self.model(data_t)

            # 1.1 Loss for content classifier on source domain
            output_content_src: torch.Tensor = self.classifier_content(feature_s)
            loss_content_src: torch.Tensor = self.loss_func(output_content_src, label_s)

            # 1.2 Loss for domain classifier on both domain
            feature_s, feature_t = self.grl(feature_s), self.grl(feature_t)  # GRL
            output_domain_src: torch.Tensor = self.classifier_domain(feature_s)
            output_domain_tgt: torch.Tensor = self.classifier_domain(feature_t)
            loss_domain: torch.Tensor = 0.5 * (
                self.loss_func(output_domain_src, label_ones) + self.loss_func(output_domain_tgt, label_zeros)
            )

            # 2. Computing total loss
            loss = loss_content_src + loss_domain

            # 3. Backwarding
            self.reset_grad()
            loss.backward()
            self.opt.step()
            self.opt_content.step()
            self.opt_domain.step()

            # 4. Updating learning rate by step; move it to self.train() if you want to update lr by epoch
            self.lr_scheduler.step()
            self.lr_scheduler_content.step()
            self.lr_scheduler_domain.step()

            # 5. Computing metrics
            num_samples += label_s.shape[0]
            num_correct_content += (output_content_src.argmax(dim=1) == label_s).sum().item()
            num_correct_domain += (output_domain_src.argmax(dim=1) == label_ones).sum().item() + (
                output_domain_tgt.argmax(dim=1) == label_zeros
            ).sum().item()
            train_loss_content_src += loss_content_src.item()

        return {
            "train_loss": metrics(train_loss_content_src / self.train_steps),  # content losss of src domain
            "train_acc": metrics(num_correct_content / num_samples, "red"),  # acc of content classifier on src domain
            "train_acc_domain": metrics(num_correct_domain / num_samples / 2),
        }

    @torch.inference_mode()  # disable autograd
    def test_epoch(self, epoch: int, dataloader_test: DataLoader) -> dict[str, metrics]:
        """Only relates to the test set of target domain."""
        # Helper variables
        num_correct_domain, num_correct_content, num_samples = 0, 0, 0
        test_loss_content = 0.0
        data: torch.Tensor
        targets: torch.Tensor

        self.model.eval()
        self.classifier_content.eval()
        self.classifier_domain.eval()
        for data, targets in self.progress(dataloader_test, epoch=epoch, test=True):
            self._y_true.append(targets.numpy())  # for plotting confusion matrix
            data, targets = data.to(self.device), targets.to(self.device)
            label_zeros = torch.zeros_like(targets, dtype=torch.long, device=self.device)  # tgt domain

            # Forwarding
            feature = self.model(data)
            output_content = self.classifier_content(feature)
            output_domain = self.classifier_domain(feature)
            # Computing metrics
            test_loss_content += self.loss_func(output_content, targets).item()
            num_samples += data.shape[0]
            predicts = torch.argmax(output_content, dim=1)
            num_correct_content += torch.sum(predicts == targets).item()
            num_correct_domain += torch.sum(torch.argmax(output_domain, dim=1) == label_zeros).item()

            self._y_pred.append(predicts.cpu().numpy())  # for plotting confusion matrix

        return {
            "test_loss": metrics(test_loss_content / len(dataloader_test)),  # 　content loss of tgt domain
            "test_acc": metrics(num_correct_content / num_samples, "green"),  # acc of content classifier on tgt domain
            "test_acc_domain": metrics(num_correct_domain / num_samples),
        }
