# -*- encoding: utf-8 -*-
'''
@File        :   _fed_trainer_base.py
@Description :   The base class of all federated trainers.
@Time        :   2023/07/11 03:53:41
@Author      :   Leo Yann
@Version     :   1.0
@Contact     :   yangliu991022@gmail.com
'''

from abc import ABC, abstractmethod
import itertools
import logging
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from torch.utils import tensorboard
import time
import rich.progress
import copy
from ._trainer_base import _color_map, plot_confusion
from utils.iter_DataLoader import IterDataLoader


class _Fed_Trainer_Base(ABC):
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
        self.confusion_path = path / 'confusion_matrix'
        self.save_period = info['save_period']
        self.min_valid_loss = np.inf
        self.min_valid_pretrain_loss = np.inf
        self.plot_confusion_flag = self.info.get('plot_confusion', False)
        self.num_clients = info['num_clients']  # number of clients
        self.num_clients_select = info['num_clients_select'] if ('num_clients_select' in info and 0 < info['num_clients_select'] <= self.num_clients) else self.num_clients  # number of clients selected for each round

        self.model = None  # must be defined in `self.__prepare_models()`
        self._y_true, self._y_pred = None, None  # temp variables for confusion matrix

        # 1. Dataloaders
        self._prepare_dataloaders(info)
        # 2. Defination and initialization of the models
        self._prepare_models(info)
        # 3. Clients and server
        self._prepare_clients_server(info)
        # loggers
        self._get_logger()  # txt logger
        self.metrics_writer = tensorboard.SummaryWriter(path / 'log')

    @abstractmethod
    def _prepare_dataloaders(self, info):
        pass

    @abstractmethod
    def _prepare_models(self, info):
        pass

    @abstractmethod
    def _prepare_clients_server(self, info):
        '''Should define `self.clients` (List) and `self.server`'''
        pass

    def _resuming_model(self, model):
        '''Note: only for variable `self.model`'''
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            self.epoch = checkpoint['epoch'] + 1
        else:
            self.epoch = 0

    @staticmethod
    def metrics_wrapper(metrics: dict, with_color=False) -> str:
        """
        It takes a dictionary of metrics and returns a string of the metrics in a nice format

        Args:
          metrics (dict): dict
          with_color: If True, the metrics will be printed with color codes (see -> `_color_map`). Defaults to False

        Returns:
          A string of the metrics
        """
        result = []
        if with_color:
            for key, value in metrics.items():
                if isinstance(value, (tuple, list)):
                    if value[-1] in _color_map:
                        if isinstance(value[0], (tuple, list)):
                            result.append(f"{_color_map[value[-1]][0]}{key}: {value[0]}{_color_map[value[-1]][1]} | ")
                        else:
                            result.append(f"{_color_map[value[-1]][0]}{key}: {value[0]:.4f}{_color_map[value[-1]][1]} | ")
                    else:
                        result.append(f"{key}: {value[0]} | ")
                else:
                    result.append(f"{key}: {value:.4f} | ")
        else:
            for key, value in metrics.items():
                if isinstance(value, (tuple, list)):
                    if value[-1] in _color_map:
                        if isinstance(value[0], (tuple, list)):
                            result.append(f"{key}: {value[0]} | ")
                        else:
                            result.append(f"{key}: {value[0]:.4f} | ")
                    else:
                        result.append(f"{key}: {value[0]} | ")
                else:
                    result.append(f"{key}: {value:.4f} | ")
        return "".join(result)


    def dictlist2dict(self, dictlist: list):
        """
        It takes a list of dictionaries and returns a dictionary of the mean of each key

        Args:
          dictlist (list): list of dictionaries

        Returns:
          A dictionary of the mean of each key
        """
        for m in dictlist:
            if m:
                keys = m.keys()
                break

        metrics, metrics_client, color = {}, {}, {}
        for dic in dictlist:
            if dic:
                for key in keys:
                    if key in dic:
                        if isinstance(dic[key], (tuple, list)):
                            metrics[key] = metrics.get(key, []) + [dic[key][0]]
                            metrics_client[key] = metrics_client.get(key, []) + [dic[key][0]]
                            color[key] = dic[key][1]
                        else:
                            metrics[key] = metrics.get(key, []) + [dic[key]]
                            metrics_client[key] = metrics_client.get(key, []) + [dic[key]]
            else:
                for key in keys:
                    metrics_client[key] = metrics_client.get(key, []) + [0.]

        metrics = {key: (np.mean(np.array(value)).tolist(), color[key]) if key in color else np.mean(value).tolist() for key, value in metrics.items()}
        metrics_client = {key: np.array(value) for key, value in metrics_client.items()}

        return metrics, metrics_client


    def train(self):  # sourcery skip: low-code-quality
        """
        Call train_epoch() and test_epoch() for each epoch and log the results.
        """
        self.metrics_train_epoch_client = [] # can save the metrics of each client for each epoch, but not save in this version
        for epoch in range(self.epoch, self.max_epoch):
            for client in self.clients:
                client.reset_dataloader()
            time_begin = time.time()
            max_step = min([client.num_batches_train for client in self.clients])
            local_step = min([client.local_train_step for client in self.clients])

            # train for one step
            client_set = np.random.choice(self.num_clients, self.num_clients_select, replace=False)
            metrics_train_epoch = []
            for _ in self.progress(range(0, max_step, local_step), epoch=epoch):
                '''1. Pushing models'''
                self.server.push_model()

                '''2. Training clients'''
                metrics_train_list_1step = [self.clients[i].local_train() if i in client_set else {} for i in range(self.num_clients)]
                metrics_train_1step, metrics_train_1step_client = self.dictlist2dict(metrics_train_list_1step)
                metrics_train_epoch.append(metrics_train_1step)
                self.metrics_train_epoch_client.append(metrics_train_1step_client)
                
                '''3. Aggregating models'''
                self.server.aggregate(client_set)


            metrics_train, _ = self.dictlist2dict(metrics_train_epoch)
            print(f'Epoch: {epoch:<4d}| {self.metrics_wrapper(metrics_train, with_color=True)}Testing...')

            '''4. Testing epoch'''
            metrics_test = self.server.test(epoch=epoch)
            time_end = time.time()
            print('\x1b\x4d'*2)  # move cursor up
            print(f'Epoch: {epoch:<4d}| {self.metrics_wrapper(metrics_train, with_color=True)}', end='')
            print(f'{self.metrics_wrapper(metrics_test, with_color=True)}time:{int(time_end - time_begin):3d}s', end='', flush=True)
            
            '''5. Logging results'''
            best = self._save_model_by_test_loss(epoch, metrics_test["test_loss"])  # need to be specified by yourself
            self.metrics_writer.add_scalar("test_acc", metrics_test["test_acc"][0],   # need to be specified by yourself
                                           global_step=epoch)
            # log to log.txt
            self.logger.info(f'Epoch: {epoch:<4d}| '
                             f'{self.metrics_wrapper(metrics_train)}{self.metrics_wrapper(metrics_test)}'
                             f'{"saving best model..." if best else ""}')
            self._epoch_end(epoch)  # Can be called at the end of each epoch
            self.epoch += 1
            

        self._train_end()  # Must be called at the end of the training

    def _epoch_end(self, epoch):
        pass

    def _train_end(self):
        """If this function is overrided, please call super()._train_end() at the end of the function."""
        self.metrics_writer.close()
        

    # @abstractmethod
    # def train_epoch(self, epoch):
    #     pass

    # @abstractmethod
    # def test_epoch(self, epoch):
    #     pass

    @staticmethod
    def _plot_confusion_matrix_impl(photo_path, labels, predicts, classes, normalize=True, title='Confusion Matrix',
                                    cmap=plt.cm.Oranges):
        FONT_SIZE = 13
        cm = confusion_matrix(labels, predicts, labels=list(range(len(classes))))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        plt.figure(figsize=(11, 9))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=FONT_SIZE)
        plt.colorbar(extend=None)
        plt.clim(0, 1)
        plt.xticks(np.arange(len(classes)), classes, rotation=25, fontsize=FONT_SIZE-2)
        plt.yticks(np.arange(len(classes)), classes, fontsize=FONT_SIZE-2)
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=FONT_SIZE,
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel(r'$True\; labels$', fontsize=FONT_SIZE)
        plt.xlabel(r'$Predicted\; labels$', fontsize=FONT_SIZE)
        plt.savefig(photo_path, format="png", bbox_inches='tight', dpi=100)

    @staticmethod
    def _get_object(module, s: str, parameter: dict):
        return getattr(module, s)(**parameter)

    def _get_logger(self):
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info(f'{time.strftime("%Y-%m-%d %p %A", time.localtime())} - '
                         f'model: {type(self.model).__name__}')  # Only log name of variable `self.model`

    def _save_model_by_test_loss(self, epoch, valid_loss) -> bool:
        flag = 0
        if valid_loss < self.min_valid_loss:
            flag = 1
            self.min_valid_loss = valid_loss
            if epoch % self.save_period == (__period := self.save_period-1):
                print(' | saving best model and checkpoint...')
                self._save_checkpoint(epoch, True)
                self._save_checkpoint(epoch, False)
            else:
                print(' | saving best model...')
                self._save_checkpoint(epoch, True)
        elif epoch % self.save_period == 0:
            print(' | saving checkpoint...')
            self._save_checkpoint(epoch, False)
        else:
            print()
        return flag

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.server.global_model.cpu().state_dict(),  # Only save parameters of variable `self.model`
            'loss_best': self.min_valid_pretrain_loss,
        }
        if save_best:
            best_path = str(self.model_path / ('model_best.pth'))
            torch.save(state, best_path)
        else:
            path = str(self.model_path / f'checkpoint-epoch{epoch}.pth')
            torch.save(state, path)


    def progress(self, dataloader, epoch, test=False, total=None):
        _progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.percentage]{task.description}"),
            rich.progress.SpinnerColumn("dots" if test else "moon", "progress.percentage", finished_text="[green]✔"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn("[progress.download]{task.percentage:>5.1f}%"),
            rich.progress.MofNCompleteColumn(),
            # TaskProgressColumn("[progress.percentage]{task.completed:>3d}/{task.total:<3d}"),
            "•[progress.remaining] ⏳",
            rich.progress.TimeRemainingColumn(),
            "•[progress.elapsed] ⏰",
            rich.progress.TimeElapsedColumn(),
            transient=True,
        )
        if total is None:
            try:
                total = len(dataloader)
            except TypeError:
                total = self.num_batches_test if test else self.num_batches_train

        with _progress:
            description = "Testing" if test else f"Epoch {epoch+1}/{self.max_epoch}"
            yield from _progress.track(
                dataloader, total=total, description=description, update_period=0.1
            )
            _progress.update(0, description=f"[green]Epoch {epoch+1:<2d}")

    # a class of client
    class Client(ABC):
        def __init__(self, id, info, model, dataloader_train, dataloader_test, device, confusion_path):
            self.id = id
            self.info = info
            self.model = model
            self.num_train_samples_per_batch = dataloader_train.batch_size
            self.num_batches_train = len(dataloader_train)
            self.dataloader_train = iter(IterDataLoader(dataloader_train))
            self.dataloader_test = dataloader_test
            self.device = device
            self.plot_confusion_flag = self.info.get('plot_confusion', False)
            self.confusion_path = confusion_path
            self.num_classes = self.info['num_classes']
            self._y_true, self._y_pred = None, None  # temp variables for confusion matrix
            self.max_epoch = self.info['epochs']
            self._prepare_opt(info)
            self._set_local_train_step(info)

        def _set_local_train_step(self, info):
            local_train_step = info['local_train_step'] if 'local_train_step' in info else -1
            local_train_epoch = info['local_train_epoch'] if 'local_train_epoch' in info else -1
            assert (local_train_step > 0) ^ (local_train_epoch > 0)
            self.local_train_step = local_train_step if local_train_step > 0 else local_train_epoch * self.num_batches_train

        @staticmethod
        def _adapt_epoch_to_step(params: dict, train_steps: int = None):
            if params.get('epoch_size', False):  # get epoch_size rather than step_size
                params['step_size'] = int(params['epoch_size'] * train_steps)
                params.pop('epoch_size')

        @staticmethod
        def _get_object(module, s: str, parameter: dict):
            return _Fed_Trainer_Base._get_object(module, s, parameter)

        def reset_dataloader(self):
            self.dataloader_train = iter(IterDataLoader(self.dataloader_train.dataloader))

        def __len__(self):
            return self.num_train_samples_per_batch * self.local_train_step if self.local_train_step % self.num_batches_train \
                                            else (self.local_train_step // self.num_batches_train) * len(self.dataloader_train.dataloader.dataset)
        
        

        @abstractmethod
        def _prepare_opt(self, info):
            pass

        @abstractmethod
        def _reset_grad(self):
            pass

        @abstractmethod
        def set_model(self, model):
            pass

        @abstractmethod
        def train_step(self):
            pass

        @abstractmethod
        def local_train(self):
            pass        

        @abstractmethod
        def local_test(self, epoch):
            pass


    # a class of server
    class Server(ABC):
        def __init__(self, clients, global_model):
            self.clients = clients
            self.global_model = global_model
            self.plot_confusion_flag = self.clients[0].plot_confusion_flag
            self.confusion_path = self.clients[0].confusion_path
            self.num_classes = self.clients[0].num_classes
            self.max_epoch = min([client.max_epoch for client in self.clients])
            self._y_true, self._y_pred = None, None  # temp variables for confusion matrix

        def push_model(self):
            for client in self.clients:
                client.set_model(self.global_model.cpu())

        @abstractmethod
        def aggregate(self, client_set):
            pass

        @abstractmethod
        def test(self, epoch):
            pass

        
# a decorator to plot confusion matrix easily
def plot_confusion_fed(name='test', interval=1, is_client=False):
    def decorator(func_to_plot_confusion):
        # wrapper to the actual function, e.g. self.test_epoch(self, epoch, *args, **kwargs)
        def wrapper(self, epoch, *args, **kwargs):
            # 1. before the func: empty the public list
            self._y_pred, self._y_true = [], []
            # 2. call the func
            metrics = func_to_plot_confusion(self, epoch, *args, **kwargs)
            # 3. after the func: check and plot the confusion matrix
            if epoch % interval == 0 and (len(self._y_pred) + len(self._y_true)) > 0:
                if isinstance(self._y_pred, list) or isinstance(self._y_true, list):
                    self._y_pred = np.concatenate(self._y_pred, axis=0)
                    self._y_true = np.concatenate(self._y_true, axis=0)
                if is_client:
                    os.makedirs(self.confusion_path / f'client-{self.id}', exist_ok=True)
                else:
                    os.makedirs(self.confusion_path / f'server', exist_ok=True)
                _Fed_Trainer_Base._plot_confusion_matrix_impl(
                    photo_path=self.confusion_path / (f'client-{self.id}' if is_client else 'server') / f'{name}-{str(epoch).zfill(len(str(self.max_epoch)))}.png',
                    labels=self._y_true, predicts=self._y_pred, classes=list(range(self.num_classes)))
            return metrics

        return wrapper

    return decorator
