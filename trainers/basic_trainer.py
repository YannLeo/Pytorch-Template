# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:20:01
# @File       : basic_trainer.py
# @Note       : A basic trainer for training a feed forward neural network


import torch
from torch import nn
import numpy as np
from pathlib import Path
import tqdm  
from ._trainer_base import _Trainer_Base
import models
import datasets


class BasicTrainer(_Trainer_Base):
    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        # Dataloaders, models, optimizers and loggers are prepared in super().__init__()
        super().__init__(info, resume, path, device)
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=info["label_smoothing"])
        
    def _prepare_dataloaders(self, info):
        """
        Prepare dataloaders for training and testing.
        """
        # dataloaders!
        self.dataset_train = self._get_object(datasets, info['dataloader_train']['dataset']['name'],
                                               info['dataloader_train']['dataset']['args'])
        self.dataloader_train = torch.utils.data.DataLoader(dataset=self.dataset_train,
                                                            **info['dataloader_train']['args'])
        self.dataset_test = self._get_object(datasets, info['dataloader_test']['dataset']['name'],
                                              info['dataloader_test']['dataset']['args'])
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test,
                                                           **info['dataloader_test']['args'])
        # helper constants
        self.batch_size = self.dataloader_train.batch_size
        self.num_batches_train = len(self.dataloader_train)
        self.num_batches_test = len(self.dataloader_test)
        
    def _prepare_models(self, info):
        """
        Prepare models for training.
        """
        # the name `self.model` is reserved for some functions in the base class
        self.model = self._get_object(models, info['model']['name'], info['model']['args'])
        self._resuming_model(self.model)  # Prepare for resuming models
        self.model = self.model.to(self.device)
    
    def _prepare_opt(self, info):
        """
        Prepare optimizers and corresponding learning rate schedulers.
        """
        self.opt = torch.optim.AdamW(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                              {'optimizer': self.opt, **info['lr_scheduler']['args']})  
    
    def _reset_grad(self):
        """
        Reset gradients of all trainable parameters.
        """
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
        loop = tqdm.tqdm(enumerate(self.dataloader_train), total=self.num_batches_train, leave=False, 
                         desc=f"Epoch {epoch}/{self.max_epoch}")
        for batch, (data, targets) in loop:
            data, targets = data.to(self.device), targets.to(self.device) 

            # 1. Forwarding
            output = self.model(data)
            # 2. Computing loss
            loss = self.loss_func(output, targets)
            # 3. Backwarding: compute gradients and update parameters
            self._reset_grad()
            loss.backward()
            self.opt.step()
            # 4. Updating learning rate by step; move it to self.train() if you want to update lr by epoch
            self.metrics_writer.add_scalar("lr", self.opt.param_groups[0]["lr"], 
                                           global_step=epoch*self.num_batches_train+batch)
            self.lr_scheduler.step()
            # 5. Computing metrics
            num_samples += data.shape[0]
            train_loss += loss.item()
            num_correct += torch.sum(output.argmax(dim=1) == targets).item()
            
            # Display at the end of the progress bar
            if batch % (__interval:=1 if self.num_batches_train > 10 else self.num_batches_train // 10) == 0:
                loop.set_postfix(loss_step=f"{loss.item():.3f}", refresh=False)

        return {
            "train_loss": train_loss / self.num_batches_train,
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
                # Forwarding
                output = self.model(data)
                # Computing metrics
                test_loss += self.loss_func(output, targets).item()
                num_samples += data.shape[0]
                predicts = output.argmax(dim=1)
                num_correct += torch.sum(predicts == targets).item()
                
                if self.plot_confusion:
                    self._y_pred.append(predicts.cpu().numpy())
                    
        return {
            "test_loss": test_loss / self.num_batches_test,
            "test_acc": (num_correct / num_samples, 'red'),  # (value, color) is supported
        }
    
    