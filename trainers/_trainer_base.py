# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:18:02
# @File       : _trainer_base.py
# @Note       : The base class of all trainers, and some tools for training

from abc import ABC, abstractmethod
import itertools
import logging
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import time

_color_map = {
    "black": ("\033[30m", "\033[0m"),  # 黑色字
    "red": ("\033[31m", "\033[0m"),  # 红色字
    "green": ("\033[32m", "\033[0m"),  # 绿色字
    "yellow": ("\033[33m", "\033[0m"),  # 黄色字
    "blue": ("\033[34m", "\033[0m"),  # 蓝色字
    "magenta": ("\033[35m", "\033[0m"),  # 紫色字
    "cyan": ("\033[36m", "\033[0m"),  # 青色字
    "white": ("\033[37m", "\033[0m"),  # 白色字
    "white_on_k": ("\033[40;37m", "\033[0m"),  # 黑底白字
    "white_on_r": ("\033[41;37m", "\033[0m"),  # 红底白字
    "white_on_g": ("\033[42;37m", "\033[0m"),  # 绿底白字
    "white_on_y": ("\033[43;37m", "\033[0m"),  # 黄底白字
    "white_on_b": ("\033[44;37m", "\033[0m"),  # 蓝底白字
    "white_on_m": ("\033[45;37m", "\033[0m"),  # 紫底白字
    "white_on_c": ("\033[46;37m", "\033[0m"),  # 天蓝白字
    "black_on_w": ("\033[47;30m", "\033[0m"),  # 白底黑字
}


class _Trainer_Base(ABC):
    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        """
        Initialize the model, optimizer, scheduler, dataloader, and logger.

        Args:
          info (dict): dict of configs from toml file
          resume: path to checkpoint
          path: the path to the folder where the model will be saved.
          device: the device to run the model on.

        ---
        Tips: This function is highly dependent on the config file, where the data, models, optimizers and 
              schedulers are defined. 
        """
        # Basic constants
        self.info = info  # dict of configs from toml file
        self.resume = resume  # path to checkpoint
        self.device = device
        self.max_epoch = info['epochs']
        self.num_classes = info['model']['args']['num_classes']
        self.log_path = path / 'log' / 'log.txt'
        self.model_path = path / 'model'
        self.confusion_path = path / 'confusion'
        self.save_period = info['save_period']
        self.min_valid_loss = np.inf
        self.min_valid_pretrain_loss = np.inf

        # 1. Dataloaders
        self.__prepare_dataloaders(info)
        # 2. Defination and initialization of the models
        self.__prepare_models(info)
        # 3. Optimizers and schedulers of the models
        self.__prepare_opt(info)
        # loggers
        self.__get_logger()  # txt logger

    @abstractmethod
    def __prepare_dataloaders(self, info):
        pass

    @abstractmethod
    def __prepare_models(self, info):
        pass

    def __resuming_model(self, model):
        '''Note: only for variable `self.model`'''
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict['model'])
            self.epoch = checkpoint['epoch'] + 1
        else:
            self.epoch = 0

    @abstractmethod
    def __prepare_opt(self, info):
        pass

    @abstractmethod
    def __reset_grad(self):
        pass

    @staticmethod
    def metrics_wrapper(metrics: dict, with_color=False) -> str:
        if not with_color:
            return "".join(f"{key}: {value:.4f} | " for key, value in metrics.items())
        else:

    def train(self):  # sourcery skip: low-code-quality
        """
        Call train_epoch() and test_epoch() for each epoch and log the results.
        """
        begin_epoch = self.epoch

        for epoch in range(begin_epoch, self.max_epoch):
            time_begin = time.time()
            print(f'epoch: {epoch + 1}\t| ', end='')

            '''Training epoch'''
            metrics_train = self.train_epoch(epoch+1)
            print(f'train_loss: {train_class_loss:.6f} | train_kl_loss: {train_kl_loss:.6f} | train_acc: {train_acc:6f} | ', end='')
            print('testing...' + '\b' * len('testing...'), end='', flush=True)

            '''Testing epoch'''
            metrics_test = self.test_epoch(epoch+1)
            time_end = time.time()
            print(
                f'test_class_loss: {test_class_loss:6f} | test_acc: {test_acc:6f} | pseudo_rate: {num_pseudo/self.num_target:.6f} | '
                f'pseudo_correct: {num_correct/(num_pseudo+1e-5):.6f} | time: {int(time_end - time_begin)}s', end='')

            '''Logging results'''
            best = self.__save_model_by_valid_loss(epoch + 1, test_class_loss)
            self.metric_writer.add_scalar("test_acc", test_acc, epoch)
            self.logger.info(
                f'epoch: {epoch + 1}\t| train_class_loss: {train_class_loss:.6f} | train_kl_loss: {train_kl_loss:.6f} | '
                f'train_acc: {train_acc:.6f} | pseudo_rate: {num_pseudo/self.num_target:.6f} | '
                f'pseudo_correct: {num_correct/(num_pseudo+1e-5):.6f} | test_class_loss: {test_class_loss:.6f} | '
                f'test_acc: {test_acc:.6f}{" | saving best model..." if best else ""}')
            self.epoch += 1

            if self.info.get('confusion', False):
                self.__plot_confusion_matrix(photo_path=self.confusion_path / f'train-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png',
                                             labels=train_labels, predicts=train_predicts, classes=list(range(self.num_classes)), normalize=True)

        self.__train_end()

    def __train_end(self):
        pass

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def test_epoch(self, epoch):
        pass

    def __plot_confusion_matrix(self, photo_path, labels, predicts, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):
        FONT_SIZE = 9
        cm = confusion_matrix(labels, predicts, labels=list(range(len(classes))))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        plt.figure(figsize=(8*2, 6*2))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, classes, fontsize=FONT_SIZE)
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE+3,
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(photo_path)

    @staticmethod
    def __get_object(module, s: str, parameter: dict):
        return getattr(module, s)(**parameter)

    def __get_logger(self):
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info(f'model: {type(self.model).__name__}')  # Only log name of variable `self.model`

    def __save_model_by_valid_loss(self, epoch, valid_loss):
        flag = 0
        if valid_loss < self.min_valid_loss:
            flag = 1
            self.min_valid_loss = valid_loss
            if epoch % self.save_period == 0:
                print(' | saving best model and checkpoint...')
                self.__save_checkpoint(epoch, True)
                self.__save_checkpoint(epoch, False)
            else:
                print(' | saving best model...')
                self.__save_checkpoint(epoch, True)
        elif epoch % self.save_period == 0:
            print(' | saving checkpoint...')
            self.__save_checkpoint(epoch, False)
        else:
            print()
        return flag

    def __save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'loss_best': self.min_valid_pretrain_loss,
        }
        if save_best:
            best_path = str(self.model_path / ('model_best.pth'))
            torch.save(state, best_path)
        else:
            path = str(self.model_path / f'checkpoint-epoch{epoch}.pth')
            torch.save(state, path)
