# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 23/01/11 ~ 22:07:25
# @File       : mcd_trainer.py
# @Note       : A simple implementation of MCD (Maximum Classifier Discrepancy for
#               Unsupervised Domain Adaptation) in PyTorch


import torch
from torch import nn
import numpy as np
from pathlib import Path
import tqdm
from ._trainer_base import _Trainer_Base
import models
import datasets


class MCDTrainer(_Trainer_Base):
    """
    A simple implementation of MCD (https://arxiv.org/abs/1712.02560). The most essential 
    part of the code are the functions train_epoch() and test_epoch().

    Ref: https://github.com/mil-tokyo/MCD_DA
    """

    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        # Dataloaders, models, optimizers and loggers are prepared in super().__init__()
        super().__init__(info, resume, path, device)

        self.loss_func = nn.CrossEntropyLoss()
        self.discrepancy_steps = info["discrepancy_steps"]
        self.discrepancy_weight = info["discrepancy_weight"]

    def _prepare_dataloaders(self, info):
        """
        Prepare the dataloaders for the source and target domains.
        """
        # datasets of source domain
        self.dataset_source = self._get_object(datasets, info['dataloader_source']['dataset']['name'],
                                               info['dataloader_source']['dataset']['args'])
        self.dataset_val = self._get_object(datasets, info['dataloader_val']['dataset']['name'],
                                            info['dataloader_val']['dataset']['args'])
        self.dataloader_source = torch.utils.data.DataLoader(dataset=self.dataset_source,
                                                             **info['dataloader_source']['args'])
        self.dataloader_val = torch.utils.data.DataLoader(dataset=self.dataset_val,
                                                          **info['dataloader_val']['args'])
        # datasets of target domain
        self.dataset_target = self._get_object(datasets, info['dataloader_target']['dataset']['name'],
                                               info['dataloader_target']['dataset']['args'])
        self.dataset_test = self._get_object(datasets, info['dataloader_test']['dataset']['name'],
                                             info['dataloader_test']['dataset']['args'])
        self.dataloader_target = torch.utils.data.DataLoader(dataset=self.dataset_target,
                                                             **info['dataloader_target']['args'])
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test,
                                                           **info['dataloader_test']['args'])
        # helper constants
        self.batch_size = self.dataloader_source.batch_size
        self.num_batches_train = min(len(self.dataloader_source), len(self.dataloader_target))
        self.num_batches_test = len(self.dataloader_test)  # we don't use the validation set (dataset_val)

    def _prepare_models(self, info):
        """
        Prepare the models.
        """
        # the name `self.model` is reserved for some functions in the base class
        self.model = self._get_object(models, info['model']['name'], info['model']['args'])
        self._resuming_model(self.model)  # Prepare for resuming models
        self.C1 = models.Classifier(
            input_dim=info['model']['args']['num_classes'], num_class=self.num_classes,
            intermediate_dim=128, layers=3)
        self.C2 = models.Classifier(
            input_dim=info['model']['args']['num_classes'], num_class=self.num_classes,
            intermediate_dim=128, layers=2)

        self.model = self.model.to(self.device)
        self.C1 = self.C1.to(self.device)
        self.C2 = self.C2.to(self.device)

    def _prepare_opt(self, info):
        """
        Prepare the optimizers and corresponding learning rate schedulers.
        """
        self.opt = torch.optim.AdamW(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                             {'optimizer': self.opt, **info['lr_scheduler']['args']})
        self.opt = torch.optim.AdamW(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.opt_C1 = torch.optim.Adam(params=self.C1.parameters(), lr=info['lr_scheduler_C']['init_lr'])
        self.opt_C2 = torch.optim.Adam(params=self.C2.parameters(), lr=info['lr_scheduler_C']['init_lr'])

        self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                             {'optimizer': self.opt, **info['lr_scheduler']['args']})
        self.lr_scheduler_C1 = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler_C']['name'],
                                                {'optimizer': self.opt_C1, **info['lr_scheduler']['args']})
        self.lr_scheduler_C2 = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler_C']['name'],
                                                {'optimizer': self.opt_C2, **info['lr_scheduler']['args']})

    def _reset_grad(self):
        """
        Reset gradients of all trainable parameters.
        """
        self.opt.zero_grad(set_to_none=True)
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
        loop = self.progress(enumerate(zip(self.dataloader_source, self.dataloader_target)), epoch=epoch)
        for batch, ((data_s, label_s), (data_t, label_t)) in loop:  # label_t is merely for metrics
            data_s, data_t = data_s.to(self.device), data_t.to(self.device)
            label_s, label_t = label_s.to(self.device), label_t.to(self.device)
            num_samples += data_s.shape[0]

            """step 1. Training on source domain only"""
            feature_s = self.model(data_s)
            output_s1 = self.C1(feature_s)
            output_s2 = self.C2(feature_s)
            loss_s1 = self.loss_func(output_s1, label_s)
            loss_s2 = self.loss_func(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            self._reset_grad()
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
            self._reset_grad()
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
                self._reset_grad()
                loss_dis.backward()
                self.opt.step()
            # Computing metrics
            num_correct_tgt += ((output_t1 + output_t2).argmax(dim=1) == label_t).sum().item()
            train_loss_discrepancy += loss_dis.item()

            # Updating learning rate by step
            self.lr_scheduler.step()
            self.lr_scheduler_C1.step()
            self.lr_scheduler_C2.step()

        return {
            "train_loss_C1": train_loss_C1 / self.num_batches_train,
            "train_loss_C2": train_loss_C2 / self.num_batches_train,
            "train_loss_dis": train_loss_discrepancy / self.num_batches_train,
            "acc_C1_s": num_correct_C1_src / num_samples,
            "acc_C2_s": num_correct_C2_src / num_samples,
            "acc_tgt": (num_correct_tgt / num_samples, 'green')
        }

    def test_epoch(self, epoch):
        """Only relates to the test set of target domain."""
        # Helper variables
        self._y_pred, self._y_true = [], []  # to plot confusion matrix of test dataset
        num_correct, num_correct_C1, num_correct_C2, num_samples = 0, 0, 0, 0
        test_loss = 0.

        self.model.eval()
        self.C1.eval()
        self.C2.eval()
        with torch.no_grad():
            for data, targets in self.progress(self.dataloader_test, epoch=epoch, test=True):
                if self.plot_confusion:
                    self._y_true.append(targets.numpy())

                data, targets = data.to(self.device), targets.to(self.device)

                # Forwarding
                feature = self.model(data)
                output1 = self.C1(feature)
                output2 = self.C2(feature)
                # Computing metrics
                num_samples += data.shape[0]
                test_loss += (self.loss_func(output1, targets) +
                              self.loss_func(output2, targets)).item() / 2
                predicts = torch.argmax(output1+output2, dim=1)  # ensemble
                num_correct += torch.sum(predicts == targets).item()
                num_correct_C1 += torch.sum(torch.argmax(output1, dim=1) == targets).item()
                num_correct_C2 += torch.sum(torch.argmax(output2, dim=1) == targets).item()

                if self.plot_confusion:
                    self._y_pred.append(predicts.cpu().numpy())

        return {
            "test_loss": test_loss / self.num_batches_test,
            "test_acc": (num_correct / num_samples, 'blue'),
            "test_acc_C1": num_correct_C1 / num_samples,
            "test_acc_C2": num_correct_C2 / num_samples,
        }

    @staticmethod
    def discrepancy(out1, out2):
        return torch.mean(torch.abs(torch.softmax(out1, dim=1) - torch.softmax(out2, dim=1)))
