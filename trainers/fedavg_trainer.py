# -*- encoding: utf-8 -*-
'''
@File        :   fedavg_trainer.py
@Description :   
@Time        :   2023/07/14 00:40:39
@Author      :   Leo Yann
@Version     :   1.0
@Contact     :   yangliu991022@gmail.com
'''

import torch
from torch import nn
import numpy as np
from pathlib import Path
import copy
from collections import OrderedDict

from utils.iter_DataLoader import IterDataLoader
from ._fed_trainer_base import _Fed_Trainer_Base, plot_confusion_fed
import models
import datasets



class FedAvgTrainer(_Fed_Trainer_Base):
    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        # Dataloaders, models, optimizers and loggers are prepared in super().__init__()
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=info["label_smoothing"])
        super().__init__(info, resume, path, device)
        

    def _prepare_dataloaders(self, info):
        assert 'clients_args' in info['dataloader_train']['dataset'] and \
               'clients_args' in info['dataloader_test']['dataset']
        
        self.dataset_train, self.dataset_test = [], []
        train_args = info['dataloader_train']['dataset']['args']
        test_args = info['dataloader_test']['dataset']['args']
        for i in range(self.num_clients):
            train_args_client = copy.deepcopy(train_args)
            for key, value in info['dataloader_train']['dataset']['clients_args'].items():
                assert len(value) == self.num_clients
                train_args_client[key] = value[i]
                self.dataset_train.append(self._get_object(datasets, info['dataloader_train']['dataset']['name'], train_args_client))
            test_args_client = copy.deepcopy(test_args)
            for key, value in info['dataloader_test']['dataset']['clients_args'].items():
                assert len(value) == self.num_clients
                test_args_client[key] = value[i]
                self.dataset_test.append(self._get_object(datasets, info['dataloader_test']['dataset']['name'], test_args_client))
        
        self.dataloader_train = [torch.utils.data.DataLoader(dataset=self.dataset_train[i],
                                                            **info['dataloader_train']['args']) for i in range(self.num_clients)]
        self.dataloader_test = [torch.utils.data.DataLoader(dataset=self.dataset_test[i],
                                                              **info['dataloader_test']['args']) for i in range(self.num_clients)]
        # helper constants
        self.batch_size = self.dataloader_train[0].batch_size
        self.num_batches_train = min([len(self.dataloader_train[i]) for i in range(self.num_clients)])
        self.num_batches_test = sum([len(self.dataloader_test[i]) for i in range(self.num_clients)])


    def _prepare_models(self, info):
        """
        Prepare models for training.
        """
        # the name `self.model` is reserved for some functions in the base class
        self.model = self._get_object(models, info['model']['name'], info['model']['args'])
        self._resuming_model(self.model)  # Prepare for resuming models

    def _prepare_clients_server(self, info):
        '''Should define `self.clients` (List) and `self.server`'''
        self.clients = [self.Client(
                            id = i, 
                            info = self.info,
                            model = copy.deepcopy(self.model.cpu()),
                            dataloader_train = self.dataloader_train[i],
                            dataloader_test = self.dataloader_test[i],
                            device = self.device,
                            confusion_path = self.confusion_path,
                            loss_func = self.loss_func,
                        )
                        for i in range(self.num_clients)]
        self.server = self.Server(
                            clients = self.clients,
                            global_model = copy.deepcopy(self.model.cpu()),
                        )

    
    class Client(_Fed_Trainer_Base.Client):
        def __init__(self, id, info, model, dataloader_train, dataloader_test, device, confusion_path, loss_func=nn.CrossEntropyLoss()):
            super().__init__(id, info, model, dataloader_train, dataloader_test, device, confusion_path)
            self.loss_func = loss_func

        def _prepare_opt(self, info):
            self._adapt_epoch_to_step(info['lr_scheduler']['args'], self.num_batches_train)
            self.opt = torch.optim.AdamW(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
            self.lr_scheduler = self._get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                                {'optimizer': self.opt, **info['lr_scheduler']['args']})  

        def _reset_grad(self):
            self.opt.zero_grad(set_to_none=True)

        def set_model(self, model):
            self.model.load_state_dict(model.cpu().state_dict())

        def train_step(self):
            data, targets = next(self.dataloader_train)
            data, targets = data.to(self.device), targets.to(self.device)
            output = self.model(data)
            loss = self.loss_func(output, targets)
            self._reset_grad()
            loss.backward()
            self.opt.step()
            self.lr_scheduler.step()
            return {
                "train_loss": loss.item(),
                "num_samples": data.shape[0], 
                "num_correct": torch.sum(output.argmax(dim=1) == targets).item()
            }
         
        def local_train(self):
            num_samples, num_correct = 0, 0
            train_loss = 0.

            self.model = self.model.to(self.device)
            self.model.train()
            for _ in range(self.local_train_step):
                metrics_train_1step = self.train_step()
                train_loss += metrics_train_1step["train_loss"]
                num_samples += metrics_train_1step["num_samples"]
                num_correct += metrics_train_1step["num_correct"]

            self.model = self.model.cpu()
            return {
                "train_loss": train_loss / self.local_train_step,
                "train_acc": (num_correct / num_samples, 'blue'),  # (value, color) is supported
            }

        @plot_confusion_fed(name="test", is_client=True)
        def local_test(self, epoch):
            num_correct, num_samples = 0, 0
            test_loss = 0.

            self.model = self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                for data, targets in self.dataloader_test:
                    if self.plot_confusion_flag:
                        self._y_true.append(targets.numpy())
                    data, targets = data.to(self.device), targets.to(self.device)
                    # Forwarding
                    output = self.model(data)
                    # Computing metrics
                    test_loss += self.loss_func(output, targets).item()
                    num_samples += data.shape[0]
                    predicts = output.argmax(dim=1)
                    num_correct += torch.sum(predicts == targets).item()

                    if self.plot_confusion_flag:
                        self._y_pred.append(predicts.cpu().numpy())

            self.model = self.model.cpu()
            return {
                "test_loss": test_loss / len(self.dataloader_test),
                "num_samples": num_samples,
                "num_correct": num_correct,
            }, {
                "y_true": self._y_true,
                "y_pred": self._y_pred,
            }
            


    # a class of server
    class Server(_Fed_Trainer_Base.Server):
        def __init__(self, clients, global_model):
            super().__init__(clients, global_model)

        def push_model(self):
            for client in self.clients:
                client.set_model(self.global_model.cpu())

        def aggregate(self, client_set):
            self.global_model = self.global_model.cpu()

            aggregate_weights = OrderedDict()
            for client in self.clients:
                if client.id in client_set:
                    aggregate_weights[client.id] = len(client)
            num_train_samples = sum(aggregate_weights.values())
            aggregate_weights = {k: v / num_train_samples for k, v in aggregate_weights.items()}

            update_state = OrderedDict()

            for client in self.clients:
                if client.id in client_set:
                    local_state = client.model.cpu().state_dict()
                    for key in self.global_model.state_dict().keys():
                        if key in update_state:
                            update_state[key] += local_state[key] * aggregate_weights[client.id]
                        else:
                            update_state[key] = local_state[key] * aggregate_weights[client.id]

            self.global_model.load_state_dict(update_state)

        @plot_confusion_fed(name="test", is_client=False)
        def test(self, epoch):
            self.push_model()
            num_correct, num_samples = 0, 0
            test_loss = 0.

            metrics_test_client, result_confusion_client = [], []
            for client in self.clients:
                metrics_test, result_confusion = client.local_test(epoch)
                metrics_test_client.append(metrics_test)
                result_confusion_client.append(result_confusion)
    
            num_correct = sum([m['num_correct'] for m in metrics_test_client])
            num_samples = sum([m['num_samples'] for m in metrics_test_client])
            weight = [m['num_samples'] / num_samples for m in metrics_test_client]
            test_loss = sum([m['test_loss'] * w for m, w in zip(metrics_test_client, weight)])

            if self.plot_confusion_flag:
                for m in result_confusion_client:
                    self._y_true += m['y_true']
                    self._y_pred += m['y_pred']

            return {
                "test_loss": test_loss,
                "test_acc": (num_correct / num_samples, 'red'),
                "test_acc_clients": ([m['num_correct'] / m['num_samples'] for m in metrics_test_client], 'cyan')
            }
                