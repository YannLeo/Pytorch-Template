# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:20:01
# @File       : basic_trainer.py
# @Note       : A basic trainer for training a feed forward neural network


import torch
from torch import nn
import numpy as np
from pathlib import Path
import tqdm  # TODO
from ._trainer_base import _Trainer_Base
import models
import datasets


class BasicTrainer(_Trainer_Base):
    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        super().__init__(info, resume, path, device)
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=info["label_smoothing"])
        
    def _prepare_dataloaders(self, info):
        # datasets!
        self.dataset_train = self._get_object(datasets, info['dataloader_train']['dataset']['name'],
                                               info['dataloader_train']['dataset']['args'])
        self.dataset_test = self._get_object(datasets, info['dataloader_test']['dataset']['name'],
                                              info['dataloader_test']['dataset']['args'])
        # helper constants
        self.num_train = len(self.dataset_train)
        self.num_test = len(self.dataset_test)
        # dataloaders!
        self.dataloader_train = torch.utils.data.DataLoader(dataset=self.dataset_train,
                                                            **info['dataloader_train']['args'])
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test,
                                                           **info['dataloader_test']['args'])
        # helper constants
        self.batch_size = self.dataloader_train.batch_size
        self.num_batch_train = len(self.dataloader_train)
        self.num_batch_test = len(self.dataloader_test)
        self.interval_display_train = 1 if self.num_batch_train < 10 else self.num_batch_train // 10  # TODO
        self.interval_display_test = 1 if self.num_batch_test < 10 else self.num_batch_test // 10
        
    def _prepare_models(self, info):
        # the name `self.model` is reserved for some functions in the base class
        self.model = self._get_object(models, info['model']['name'], info['model']['args'])
        self._resuming_model(self.model)  # Prepare for resuming models
        self.model = self.model.to(self.device)
    
    def _prepare_opt(self, info):
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                              {'optimizer': self.opt, **info['lr_scheduler']['args']})  
    
    def _reset_grad(self):
        self.opt.zero_grad(set_to_none=True)

    '''
    def train(self):  # You may need to override this function to customize your training process
        pass
    '''
    
    def train_epoch(self, epoch):  # sourcery skip: low-code-quality
        """
        The main training process
        """
        # helper variables
        num_samples, num_correct = 0, 0  
        train_loss = 0. 

        self.model.train()  # don't forget
        for batch, (data, targets) in enumerate(self.dataloader_train):
            data, targets = data.to(self.device), targets.to(self.device) 

            # 1. forwarding
            predicts = self.model(data)
            # 2. computing loss
            loss = self.loss_func(predicts, targets)
            # 3. backwarding: compute gradients and update parameters
            self._reset_grad()
            loss.backward()
            self.opt.step()
            
            # 4. updating learning rate by step; move it to self.train() if you want to update lr by epoch
            self.metrics_writer.add_scalar("lr", self.opt.param_groups[0]["lr"], epoch*self.num_batch_train+batch)
            self.lr_scheduler.step()
            
            # 5. computing metrics
            num_samples += data.shape[0]
            train_loss += loss.item()
            num_correct += torch.sum(predicts.argmax(dim=1) == targets).item()

            if batch % self.interval_display_train == 0:
                print('training... batch: {}/{} | loss: {:6f}'.format(batch, self.num_batch_train, loss.item()) +
                      '\b' * len('training... batch: {}/{} | loss: {:6f}'.format(batch, self.num_batch_train, loss.item())), end='', flush=True)

        return {
            "train_loss": train_loss / self.num_batch_train,
            "train_acc": (num_correct / num_samples, 'blue'),  # (value, color) is supported
        }

    def test_epoch(self, epoch):
        self._y_pred, self._y_true = [], []  
        num_correct, num_samples = 0, 0
        test_loss = 0.
        
        self.model.eval()  # don't forget
        with torch.no_grad():
            for data, targets in self.dataloader_test:
                if self.plot_confusion:
                    self._y_true.append(targets.numpy())
                    
                data, targets = data.to(self.device), targets.to(self.device)
                num_samples += data.shape[0]
                # forwarding
                predicts = self.model(data)
                # computing metrics
                test_loss += self.loss_func(predicts, targets).item()
                pred_labels = predicts.argmax(dim=1)
                num_correct += torch.sum(pred_labels == targets).item()
                
                if self.plot_confusion:
                    self._y_pred.append(pred_labels.cpu().numpy())
                    
        return {
            "test_loss": test_loss / self.num_batch_test,
            "test_acc": (num_correct / num_samples, 'red'),  # (value, color) is supported
        }
    
    