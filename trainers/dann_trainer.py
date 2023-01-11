# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 23/01/07 ~ 17:02:47
# @File       : dann_trainer.py
# @Note       : A simple implementation of DANN (Domain-Adversarial Training of Neural Networks)
#               in PyTorch


import torch
from torch import nn
import numpy as np
from pathlib import Path
import tqdm
from ._trainer_base import _Trainer_Base
import models
import datasets


class _GradientReverseFunction(torch.autograd.Function):
    """
    Helper class for gradient reversal layer (_GradientReverseLayer)
    """
    @staticmethod
    def forward(ctx, input, coeff=1.) -> torch.Tensor:
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None


class _GradientReverseLayer(nn.Module):
    """
    Ref: https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/grl.py

    Args:
      coeff: The coefficient of the reversed gradient, and must be a positive number.
    """

    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return _GradientReverseFunction.apply(x, self.coeff)


class DANNTrainer(_Trainer_Base):
    """
    A simple implementation of DANN (https://arxiv.org/abs/1505.07818) in PyTorch. The 
    most essential part of the code are the functions train_epoch() and test_epoch().
    """

    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        # Dataloaders, models, optimizers and loggers are prepared in super().__init__()
        super().__init__(info, resume, path, device)
        
        self.loss_func = nn.CrossEntropyLoss()
        self.grl = _GradientReverseLayer(coeff=info['GRL_coeff'])

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
        self.classifier_content = models.Classifier(
            input_dim=info['model']['args']['num_classes'], num_class=self.num_classes,
            intermediate_dim=128, layers=2)
        self.classifier_domain = models.Classifier(
            input_dim=info['model']['args']['num_classes'], num_class=self.num_classes,
            intermediate_dim=128, layers=2)

        self.model = self.model.to(self.device)
        self.classifier_content = self.classifier_content.to(self.device)
        self.classifier_domain = self.classifier_domain.to(self.device)

    def _prepare_opt(self, info):
        """
        Prepare the optimizers and corresponding learning rate schedulers.
        """
        self.opt = torch.optim.AdamW(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                             {'optimizer': self.opt, **info['lr_scheduler']['args']})
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.opt_content = torch.optim.Adam(params=self.classifier_content.parameters(), lr=info['lr_scheduler']['init_lr']/2)
        self.opt_domain = torch.optim.Adam(params=self.classifier_domain.parameters(), lr=info['lr_scheduler']['init_lr']/2)

        self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                             {'optimizer': self.opt, **info['lr_scheduler']['args']})
        self.lr_scheduler_content = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                                     {'optimizer': self.opt_content, **info['lr_scheduler']['args']})
        self.lr_scheduler_domain = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                                    {'optimizer': self.opt_domain, **info['lr_scheduler']['args']})

    def _reset_grad(self):
        """
        Reset gradients of all trainable parameters.
        """
        self.opt.zero_grad(set_to_none=True)
        self.opt_content.zero_grad(set_to_none=True)
        self.opt_domain.zero_grad(set_to_none=True)

    def train_epoch(self, epoch):  # sourcery skip: low-code-quality
        """
        Main training process
        """
        # Helper variables
        train_loss_content_src = 0
        num_samples = 0  # source domain only
        num_correct_content, num_correct_domain = 0, 0
        # Domain labels
        label_ones = torch.ones(self.batch_size).long().to(self.device)  # src domain
        label_zeros = torch.zeros(self.batch_size).long().to(self.device)  # tgt domain

        self.model.train()
        self.classifier_content.train()
        self.classifier_domain.train()
        loop = tqdm.tqdm(enumerate(zip(self.dataloader_source, self.dataloader_target)),
                         total=self.num_batches_train, leave=False,
                         desc=f"Epoch {epoch}/{self.max_epoch}")
        for batch, ((data_s, label_s), (data_t, _)) in loop:  # we do not use label_t here
            data_s, data_t, label_s = data_s.to(self.device), data_t.to(self.device), label_s.to(self.device)

            feature_s = self.model(data_s)
            feature_t = self.model(data_t)

            # 1.1 Loss for content classifier on source domain
            output_content_src = self.classifier_content(feature_s)
            loss_content_src = self.loss_func(output_content_src, label_s)

            # 1.2 Loss for domain classifier on both domain
            feature_s, feature_t = self.grl(feature_s), self.grl(feature_t)  # GRL
            output_domain_src = self.classifier_domain(feature_s)
            output_domain_tgt = self.classifier_domain(feature_t)
            loss_domain = 0.5 * (self.loss_func(output_domain_src, label_ones) +
                                 self.loss_func(output_domain_tgt, label_zeros))

            # 2. Computing total loss
            loss = loss_content_src + loss_domain

            # 3. Backwarding
            self._reset_grad()
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
            num_correct_domain += (output_domain_src.argmax(dim=1) == label_ones).sum().item() + \
                (output_domain_tgt.argmax(dim=1) == label_zeros).sum().item()
            train_loss_content_src += loss_content_src.item()

            # Display at the end of the progress bar
            if batch % (__interval := 1 if self.num_batches_train > 10 else self.num_batches_train // 10) == 0:
                loop.set_postfix(loss_step=f"{loss_content_src.item():.3f}", refresh=False)

        return {
            "train_loss": train_loss_content_src / self.num_batches_train,  # content losss of src domain
            "train_acc": (num_correct_content / num_samples, 'red'),  # acc of content classifier on src domain
            "train_acc_domain": num_correct_domain / num_samples / 2,
        }

    def test_epoch(self, epoch):
        """Only relates to the test set of target domain."""
        # Helper variables
        self._y_pred, self._y_true = [], []  # to plot confusion matrix of test dataset
        num_correct_domain, num_correct_content, num_samples = 0, 0, 0
        test_loss_content = 0.
        label_zeros = torch.zeros(self.batch_size).long().to(self.device)  # tgt domain

        self.model.eval()
        self.classifier_content.eval()
        self.classifier_domain.eval()
        with torch.no_grad():
            for data, targets in self.dataloader_test:
                if self.plot_confusion:
                    self._y_true.append(targets.numpy())

                data, targets = data.to(self.device), targets.to(self.device)
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

                if self.plot_confusion:
                    self._y_pred.append(predicts.cpu().numpy())

        return {
            "test_loss": test_loss_content / self.num_batches_test,  # 　content loss of tgt domain
            "test_acc": (num_correct_content / num_samples, 'green'),  # acc of content classifier on tgt domain
            "test_acc_domain": num_correct_domain / num_samples,
        }
