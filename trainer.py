import torch
import dataset
import model
from pathlib import Path
from numpy import inf

class BaseTrainer():
    def __init__(self, info, resume=None, path=Path(), device=torch.device('cuda')):
        self.info = info
        self.resume = resume
        self.device = device
        self.max_epoch = info['epoch']
        self.dataset_train = self.get_object(dataset, info['dataloader_train']['dataset']['type'], info['dataloader_train']['dataset']['args'])
        self.dataset_test = self.get_object(dataset, info['dataloader_test']['dataset']['type'], info['dataloader_test']['dataset']['args'])
        self.dataloader_train = torch.utils.data.Dataloader(dataset=self.dataset_train, **info['dataloader_train']['args'])
        self.dataloader_test = torch.utils.data.Dataloader(dataset=self.dataset_train, **info['dataloader_train']['args'])
        self.critern = self.get_object(torch.nn, info['critern'], dict())
        self.model = self.get_object(model, info['model']['type'], info['model']['args'])
        self.model = self.model.to(self.device)
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
            self.epoch = checkpoint['epoch'] + 1
        else:
            self.epoch = 0
        self.optimizer = self.get_object(torch.optim, info['optimizer']['type'], {'params':self.model.parameters(), **info['optimizer']['args']})
        self.lr_scheduler = self.get_object(torch.optim.lr_scheduler, info['lr_scheduler']['type'], {'optimizer':self.optimizer, **info['lr_scheduler']['args']})
        self.log_path = path / 'log.txt'
        self.model_path = path
        self.save_period = info['save_period']


    def get_object(self, module, s:str, parameter:dict):
        return getattr(module, s)(**parameter)

    
    def train(self):
        start_epoch = self.epoch
        self.min_valid_loss = inf
        num_train = len(self.dataset_train)
        num_test = len(self.dataset_test)
        num_train_batch = num_train // self.dataloader_train.batch_size
        num_test_batch = num_test // self.dataloader_test.batch_size
        for epoch in range(start_epoch, self.max_epoch):
            
            self.train_epoch(epoch + 1)
            self.valid_epoch(epoch + 1)

    def train_epoch(self, epoch):
        print('epoch: {} \t| '.format(epoch), end='')

    def valid_epoch(self, epoch):
        pass