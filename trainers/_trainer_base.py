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
from torch.utils import tensorboard
import time


# Used to print the log in different colors: r, g, b, w, c, m, y, k
_color_map = {
    "black":      ("\033[30m", "\033[0m"),  # 黑色字
    "k":          ("\033[30m", "\033[0m"),  # 黑色字
    "red":        ("\033[31m", "\033[0m"),  # 红色字
    "r":          ("\033[31m", "\033[0m"),  # 红色字
    "green":      ("\033[32m", "\033[0m"),  # 绿色字
    "g":          ("\033[32m", "\033[0m"),  # 绿色字
    "yellow":     ("\033[33m", "\033[0m"),  # 黄色字
    "y":          ("\033[33m", "\033[0m"),  # 黄色字
    "blue":       ("\033[34m", "\033[0m"),  # 蓝色字
    "b":          ("\033[34m", "\033[0m"),  # 蓝色字
    "magenta":    ("\033[35m", "\033[0m"),  # 紫色字
    "m":          ("\033[35m", "\033[0m"),  # 紫色字
    "cyan":       ("\033[36m", "\033[0m"),  # 青色字
    "c":          ("\033[36m", "\033[0m"),  # 青色字
    "white":      ("\033[37m", "\033[0m"),  # 白色字
    "w":          ("\033[37m", "\033[0m"),  # 白色字
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
        self.num_classes = info['num_classes']
        self.log_path = path / 'log' / 'log.txt'
        self.model_path = path / 'model'
        self.confusion_path = path / 'confusion'
        self.save_period = info['save_period']
        self.min_valid_loss = np.inf
        self.min_valid_pretrain_loss = np.inf
        self.plot_confusion = self.info.get('plot_confusion', False)

        self.model = None  # must be defined in `self.__prepare_models()`
        self._y_true, self._y_pred = None, None  # temp variables for confusion matrix

        # 1. Dataloaders
        self.__prepare_dataloaders(info)
        # 2. Defination and initialization of the models
        self.__prepare_models(info)
        # 3. Optimizers and schedulers of the models
        self.__prepare_opt(info)
        # loggers
        self.__get_logger()  # txt logger
        self.metrics_writer = tensorboard.SummaryWriter(path / 'log') 

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
        return (
            "".join(
                f"{_color_map[value[1]][0]}{key}: {value[0]:.4f}{_color_map[value[1]][1]} | "
                if isinstance(value, (tuple, list))
                else f"{key}: {value:.4f} | "
                for key, value in metrics.items()
            )
            if with_color
            else "".join(
                f"{key}: {value[0] if isinstance(value, (tuple, list)) else value:.4f} | "
                for key, value in metrics.items()
            )
        )

    def train(self):  # sourcery skip: low-code-quality
        """
        Call train_epoch() and test_epoch() for each epoch and log the results.
        """
        for epoch in range(self.epoch, self.max_epoch):
            time_begin = time.time()
            print(f'epoch: {epoch}\t| ', end='')

            '''1. Training epoch'''
            metrics_train = self.train_epoch(epoch+1)
            print(self.metrics_wrapper(metrics_train, with_color=True), end='')
            print('testing...' + '\b' * len('testing...'), end='', flush=True)

            '''Testing epoch'''
            metrics_test = self.test_epoch(epoch+1)
            time_end = time.time()
            print(f'{self.metrics_wrapper(metrics_test, with_color=True)}time: {int(time_end - time_begin)}s', end='')

            '''Logging results'''
            best = self.__save_model_by_test_loss(epoch, metrics_test["test_loss"])  # need to be specified by yourself
            self.metrics_writer.add_scalar("test_acc", metrics_test["test_acc"], epoch)  # need to be specified by yourself
            # log to log.txt
            self.logger.info(f'epoch: {epoch}\t| '
                             f'{self.metrics_wrapper(metrics_train)}{self.metrics_wrapper(metrics_test)}'
                             f'{"saving best model..." if best else ""}')
            self.__epoch_end(epoch)  # Can be called at the end of each epoch
            self.epoch += 1

        self.__train_end()  # Must be called at the end of the training

    def __epoch_end(self, epoch):
        """If this function is overrided, please call super().__epoch_end() at the end of the function."""
        if self.plot_confusion and self._y_pred and self._y_true:
            self.__plot_confusion_matrix(
                photo_path=self.confusion_path / f'test-{str(epoch).zfill(len(str(self.max_epoch)))}.png',
                labels=self._y_true, predicts=self._y_pred, classes=list(range(self.num_classes)))
    
    def __train_end(self):
        """If this function is overrided, please call super().__epoch_end() at the end of the function."""
        self.metrics_writer.close()

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def test_epoch(self, epoch):
        pass

    @staticmethod
    def __plot_confusion_matrix(photo_path, labels, predicts, classes, normalize=True, title='Confusion Matrix',
                                cmap=plt.cm.Oranges):
        FONT_SIZE = 10
        cm = confusion_matrix(labels, predicts, labels=list(range(len(classes))))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        plt.figure(figsize=(8*2, 6*2))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.xticks(np.arange(len(classes)), classes, rotation=45, fontsize=FONT_SIZE)
        plt.yticks(np.arange(len(classes)), classes, fontsize=FONT_SIZE)
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE+4,
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.savefig(photo_path,
                    # format="eps", bbox_inches='tight', pad_inches=0, dpi=300,
                    )

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

    def __save_model_by_test_loss(self, epoch, valid_loss):
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
            'state_dict': self.model.state_dict(),  # Only save parameters of variable `self.model`
            'loss_best': self.min_valid_pretrain_loss,
        }
        if save_best:
            best_path = str(self.model_path / ('model_best.pth'))
            torch.save(state, best_path)
        else:
            path = str(self.model_path / f'checkpoint-epoch{epoch}.pth')
            torch.save(state, path)
