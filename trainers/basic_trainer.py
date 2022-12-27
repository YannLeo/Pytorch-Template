# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:20:01
# @File       : basic_trainer.py
# @Note       : A basic trainer for training a feed forward neural network

import time
import itertools
import logging
import torch
from torch import nn
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from _trainer_base import _Trainer_Base
import models
import datasets


class BasicTrainer(_Trainer_Base):
    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        super().__init__(info, resume, path, device)
        # Tensorboard logger
        self.metric_writer = tensorboard.SummaryWriter(path / 'log')  

    def __prepare_dataloaders(self, info):
        # datasets!
        self.dataset_train = self.__get_object(datasets, info['dataloader_train']['dataset']['name'],
                                               info['dataloader_train']['dataset']['args'])
        self.dataset_test = self.__get_object(datasets, info['dataloader_test']['dataset']['name'],
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
        self.num_train_batch = len(self.dataloader_train)
        self.num_test_batch = len(self.dataloader_test)
        self.train_display_interval = 1 if self.num_train_batch < 10 else self.num_train_batch // 10
        self.test_display_interval = 1 if self.num_test_batch < 10 else self.num_test_batch // 10
        
    def __prepare_models(self, info):
        self.model = self.__get_object(models, info['model']['name'], info['model']['args'])
        self.__resuming_model(self.model)  # Prepare for resuming models
        self.model = self.model.to(self.device)
    
    def __prepare_opt(self, info):
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self.__get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                              {'optimizer': self.opt, **info['lr_scheduler']['args']})  
    
    def __reset_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def __train_end(self):
        self.metric_writer.close()

    def train_epoch(self, epoch):  # sourcery skip: low-code-quality
        """
        Main training process
        """
        train_loss, train_kl_loss = 0, 0
        train_acc_num = 0
        train_num = 0
        predict, labels = [], []
        pseudo_labels = torch.ones(len(self.dataset_target)).long() * (-1)
        predicted_labels = torch.ones(len(self.dataset_target)).long() * (-1)

        self.model.train()
        self.mine.train()
        for batch, data_pack in enumerate(zip(self.dataloader_source, self.dataloader_target)):
            # `index_t` records the index of target data in the whole dataset, for pseudo labeling.
            # We do not use label_t here.
            (data_s, label_s, _), (data_t, _, index_t) = data_pack
            data_s, data_t, label_s = data_s.to(self.device), data_t.to(self.device), label_s.to(self.device)

            self.__reset_grad()
            output_s, feat_s = self.model(data_s)
            output_t, feat_t = self.model(data_t)

            """1. MINE"""
            if self.info["use_mine"] and (epoch > self.info["start_epoch"]):
                for _ in range(self.info['mine_steps']):
                    kl, ma_et, loss_kl = self.learn_KL(feat_s.detach(), feat_t.detach())
                    # gp = self.compute_gradient_penalty(feat_s.detach(), feat_t.detach())
                    self.__reset_grad()
                    loss = - 0.5 * loss_kl  # + 1.0 * gp
                    loss.backward()
                    self.opt_MINE.step()

                *_, kl_loss = self.learn_KL(feat_s, feat_t, ma_et)
                kl_loss = self.info["kl_weight"] * kl_loss
            else:
                kl_loss = torch.tensor(0)

            """2. on target domain"""
            predicted_labels = self.set_pred_labels(output_t, predicted_labels, index_t)
            if (self.info['use_class_weight']) and (epoch > self.info["start_epoch"]):
                class_weight = self.get_class_weight(predicted_labels)
            else:
                class_weight = torch.ones(self.num_classes, dtype=torch.float32).to(self.device)

            loss_content_tgt = torch.tensor(0.0)
            if self.info["use_pseudo"] and (epoch > self.info["start_epoch"]):
                output_selected, label_selected, pseudo_labels = \
                    self.set_pseudo_labels(output_t, pseudo_labels, index_t, self.info["pseudo_threshold"])
                if len(output_selected) > 0:
                    loss_content_tgt = nn.CrossEntropyLoss(weight=class_weight)(output_selected, label_selected) * 0.5

            """3. on source domain"""
            if (self.info['use_class_weight']) and (epoch > self.info["start_epoch"]):
                loss_content_src = nn.CrossEntropyLoss(weight=class_weight)(output_s, label_s) * 0.5
            else:  # pretraining
                loss_content_src = nn.CrossEntropyLoss(label_smoothing=self.info["label_smoothing"])(output_s, label_s)

            """4. total loss"""
            loss = loss_content_src + loss_content_tgt + kl_loss

            self.__reset_grad()
            loss.backward()
            self.opt.step()
            # logging the learning rate
            self.metric_writer.add_scalar("lr", self.opt.param_groups[0]["lr"], epoch*self.num_train_batch+batch)

            predict.append(torch.argmax(output_s, dim=1).cpu().detach().numpy())
            labels.append(label_s.cpu().detach().numpy())
            train_num += data_s.shape[0]
            train_loss += loss.item()
            train_kl_loss += kl_loss.item()
            train_acc_num += np.sum(predict[-1] == labels[-1])

            self.lr_scheduler.step()

            if batch % self.train_display_interval == 0:
                print('training... batch: {}/{} | total_loss: {:6f} | kl_loss: {:6f}'.format(batch, self.num_train_batch, loss.item(), kl_loss.item()) +
                      '\b' * len('training... batch: {}/{} | total_loss: {:6f} | kl_loss: {:6f}'.format(batch, self.num_train_batch, loss.item(), kl_loss.item())), end='', flush=True)

        predict = np.concatenate(predict, axis=0)
        labels = np.concatenate(labels, axis=0)
        return train_loss / self.num_train_batch, train_kl_loss / self.num_train_batch, train_acc_num / train_num, \
            predict, labels, pseudo_labels, predicted_labels

    def test_epoch(self, epoch):
        # on target domain
        test_class_loss = 0
        test_acc_num = 0
        test_num = 0
        predict, label = [], []
        self.model.eval()
        self.mine.eval()
        with torch.no_grad():
            for data, targets, _ in self.dataloader_test:
                data, targets = data.to(self.device), targets.to(self.device)
                label.append(targets.cpu().detach().numpy())
                test_num += data.shape[0]

                output, _ = self.model(data)

                loss = nn.CrossEntropyLoss()(output, targets)
                predict.append(torch.argmax(output, dim=1).cpu().detach().numpy())
                test_acc_num += torch.sum(torch.argmax(output, dim=1) == targets).item()
                test_class_loss += loss.item()

        predict = np.concatenate(predict, axis=0)
        label = np.concatenate(label, axis=0)
        return test_class_loss / self.num_test_batch, test_acc_num / test_num, predict, label
    
    
    