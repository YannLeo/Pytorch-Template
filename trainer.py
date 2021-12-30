import torch
import dataset
import model
from pathlib import Path
from numpy import inf
import logging
import time

class BaseTrainer():
    def __init__(self, info, resume=None, path=Path(), device=torch.device('cuda')):
        self.info = info
        self.resume = resume
        self.device = device
        self.max_epoch = info['epoch']
        self.dataset_train = self.get_object(dataset, info['dataloader_train']['dataset']['type'], info['dataloader_train']['dataset']['args'])
        self.dataset_test = self.get_object(dataset, info['dataloader_test']['dataset']['type'], info['dataloader_test']['dataset']['args'])
        self.dataloader_train = torch.utils.data.DataLoader(dataset=self.dataset_train, **info['dataloader_train']['args'])
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test, **info['dataloader_test']['args'])
        self.critern = self.get_object(torch.nn, info['critern'], dict())
        self.model = self.get_object(model, info['model']['type'], info['model']['args'])
        self.model = self.model.to(self.device)
        self.optimizer = self.get_object(torch.optim, info['optimizer']['type'], {'params':self.model.parameters(), **info['optimizer']['args']})
        self.lr_scheduler = self.get_object(torch.optim.lr_scheduler, info['lr_scheduler']['type'], {'optimizer':self.optimizer, **info['lr_scheduler']['args']})
        self.log_path = path / 'log' / 'log.txt'
        self.model_path = path / 'model'
        self.save_period = info['save_period']
        self.min_valid_loss = inf
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
            self.epoch = checkpoint['epoch'] + 1
        else:
            self.epoch = 0
        self.get_logger()
        self.logger.info('model: {}'.format(type(self.model).__name__))


    def get_object(self, module, s:str, parameter:dict):
        return getattr(module, s)(**parameter)

    
    def train(self):
        start_epoch = self.epoch
        self.num_train = len(self.dataset_train)
        self.num_test = len(self.dataset_test)
        self.num_train_batch = self.num_train // self.dataloader_train.batch_size
        self.num_test_batch = self.num_test // self.dataloader_test.batch_size
        self.train_display = 1 if self.num_train_batch < 10 else self.num_train_batch // 10
        self.test_display = 1 if self.num_test_batch < 10 else self.num_test_batch // 10
        for epoch in range(start_epoch, self.max_epoch):
            time_begin = time.time()
            print('epoch: {}\t| '.format(epoch + 1), end='')
            train_loss, train_acc = self.train_epoch(epoch + 1)
            print('train_loss: {:.6f} | train_acc: {:.6f} | '.format(train_loss, train_acc), end='')
            print('testing...' + '\b' * len('testing...'), end='', flush=True)
            valid_loss, valid_acc = self.valid_epoch(epoch + 1)
            time_end = time.time()
            print('valid_loss: {:.6f} | valid_acc: {:.6f} | time: {:d}s'.format(valid_loss, valid_acc, int(time_end-time_begin)), end='')
            best = self.save_model_by_valid_loss(epoch + 1, valid_loss)
            self.lr_scheduler.step()
            self.logger.info('epoch: {} \t| train_loss: {:.6f} | train_acc: {:.6f} | valid_loss: {:.6f} | valid_acc: {:.6f}'.format(
                                epoch+1, train_loss, train_acc, valid_loss, valid_acc) 
                                + ' | saving best model...' if best else '')
            self.epoch += 1
            

    def train_epoch(self, epoch):
        train_loss = 0
        train_acc_num = 0
        self.model.train()
        for batch, (data, target) in enumerate(self.dataloader_train):
            data, target = data.to(self.device), target.to(self.device)
            out = self.model(data)
            loss = self.critern(out, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc_num += torch.sum(torch.argmax(out, dim=1) == target).item()
            if batch % self.train_display == 0:
                print('training... batch: {}/{} Loss: {:.6f}'.format(batch, self.num_train_batch, loss.item()) + 
                      '\b' * len('training... batch: {}/{} Loss: {:.6f}'.format(batch, self.num_train_batch, loss.item())), end='', flush=True)
        return train_loss / self.num_train_batch, train_acc_num / self.num_train
        


    def valid_epoch(self, epoch):
        test_loss = 0
        test_acc_num = 0
        self.model.eval()
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.dataloader_test):
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                loss = self.critern(out, target)
                test_loss += loss.item()
                test_acc_num += torch.sum(torch.argmax(out, dim=1) == target).item()  
        return test_loss / self.num_test_batch, test_acc_num / self.num_test
        

    def save_model_by_valid_loss(self, epoch, test_loss):
        flag = 0
        if test_loss < self.min_valid_loss:
            flag = 1
            if epoch % self.save_period == 0:
                print(' | saving best model and checkpoint...')
                self.save_checkpoint(epoch, True)
                self.save_checkpoint(epoch, False)
            else:
                print(' | saving best model...')
                self.save_checkpoint(epoch, True)
        else:
            if epoch % self.save_period == 0:
                print(' | saving checkpoint...')
                self.save_checkpoint(epoch, False)
            else:
                print()
        self.min_valid_loss = min(test_loss, self.min_valid_loss)
        return flag


    def save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_best': self.min_valid_loss,
            'info': self.info
        }
        if save_best:
            best_path = str(self.model_path / 'model_best.pth')
            torch.save(state, best_path)
        else:
            filename = str(self.model_path / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

    def resume_checkpoint(self):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(self.resume)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.min_valid_loss = checkpoint['loss_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def get_logger(self):
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        