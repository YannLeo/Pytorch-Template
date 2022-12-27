import itertools
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from torch.utils import tensorboard
import datasets
import models


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, coeff=1.) -> torch.Tensor:
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/grl.py
    """

    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.coeff)


class DANNTrainer:
    """
    https://arxiv.org/abs/1505.07818
    """

    def __init__(self, info: dict, resume=None, path=Path(), device=torch.device('cuda')):
        # Basic variables
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
        self.criterion = nn.CrossEntropyLoss()
        # Dataloaders
        self.__prepare_dataloaders(info)
        # Defination of models
        self.model = self.__get_object(models, info['model']['name'], info['model']['args'])
        self.classifier_content = models.Classifier(in_dim=info['model']['args']['out_dim'], num_class=self.num_classes, layers=2)
        self.classifier_domain = models.Classifier(in_dim=info['model']['args']['out_dim'], num_class=self.num_classes, layers=3)
        self.grl = GradientReverseLayer(coeff=info['GRL_coeff'])
        # Optimizers and schedulers
        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.opt_content = torch.optim.Adam(params=self.classifier_content.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.opt_domain = torch.optim.Adam(params=self.classifier_domain.parameters(), lr=info['lr_scheduler']['init_lr'])
        self.lr_scheduler = self.__get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                              {'optimizer': self.opt, **info['lr_scheduler']['args']})
        self.lr_scheduler_content = self.__get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                                      {'optimizer': self.opt_content, **info['lr_scheduler']['args']})
        self.lr_scheduler_domain = self.__get_object(torch.optim.lr_scheduler, info['lr_scheduler']['name'],
                                                     {'optimizer': self.opt_domain, **info['lr_scheduler']['args']})
        # Prepare for resuming models
        self.__resuming_models()
        self.model = self.model.to(self.device)
        self.classifier_content = self.classifier_content.to(self.device)
        self.classifier_domain = self.classifier_domain.to(self.device)

        # loggers
        self.__get_logger()  # txt logger
        self.metric_writer = tensorboard.SummaryWriter(path / 'log')  # tensorboard logger

    def __prepare_dataloaders(self, info):
        self.dataset_source = self.__get_object(datasets, info['dataloader_source']['dataset']['name'],
                                                info['dataloader_source']['dataset']['args'])
        self.dataset_target = self.__get_object(datasets, info['dataloader_target']['dataset']['name'],
                                                info['dataloader_target']['dataset']['args'])
        self.dataset_valid = self.__get_object(datasets, info['dataloader_valid']['dataset']['name'],
                                               info['dataloader_valid']['dataset']['args'])
        self.dataset_test = self.__get_object(datasets, info['dataloader_test']['dataset']['name'],
                                              info['dataloader_test']['dataset']['args'])
        self.dataloader_source = torch.utils.data.DataLoader(dataset=self.dataset_source,
                                                             **info['dataloader_source']['args'])
        self.dataloader_target = torch.utils.data.DataLoader(dataset=self.dataset_target,
                                                             **info['dataloader_target']['args'])
        self.dataloader_valid = torch.utils.data.DataLoader(dataset=self.dataset_valid,
                                                            **info['dataloader_valid']['args'])
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test,
                                                           **info['dataloader_test']['args'])

    def __resuming_models(self):
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict['model'])
            self.epoch = checkpoint['epoch'] + 1
        else:
            self.epoch = 0

    def __reset_grad(self):
        self.opt.zero_grad()
        self.opt_content.zero_grad()
        self.opt_domain.zero_grad()

    def train(self):  # sourcery skip: low-code-quality
        """
        Call train_epoch() and test_epoch() for each epoch and log the results.
        """
        self.batch_size = min(self.dataloader_source.batch_size, self.dataloader_target.batch_size)
        self.num_train = len(self.dataset_source)
        self.num_target = len(self.dataset_target)
        self.num_test = len(self.dataset_test)
        self.num_train_batch = min(self.num_train // self.dataloader_source.batch_size,
                                   self.num_target // self.dataloader_target.batch_size)
        self.num_test_batch = self.num_test // self.dataloader_test.batch_size
        self.train_display = 1 if self.num_train_batch < 10 else self.num_train_batch // 10
        self.test_display = 1 if self.num_test_batch < 10 else self.num_test_batch // 10
        begin_epoch = self.epoch

        for epoch in range(begin_epoch, self.max_epoch):
            time_begin = time.time()
            print(f'epoch: {epoch + 1}\t| ', end='')

            '''Training epoch'''
            train_loss_content, train_acc_content, train_acc_domain, train_labels, train_predicts = self.train_epoch(epoch + 1)
            print(f'train_loss_content: {train_loss_content:.6f} | train_acc_content: {train_acc_content:.6f} | '
                  f'train_acc_domain: {train_acc_domain:6f} | ', end='')
            print('testing...' + '\b' * len('testing...'), end='', flush=True)

            '''Testing epoch'''
            test_loss_content, test_acc_content, test_acc_domain, test_predicts, test_labels = self.test_epoch(epoch+1)
            time_end = time.time()
            print(
                f'test_loss_content: {test_loss_content:6f} | test_acc_content: {test_acc_content:6f} | test_acc_domain: {test_acc_domain:.6f} | '
                f'time: {int(time_end - time_begin)}s', end='')

            '''Logging results'''
            best = self.__save_model_by_valid_loss(epoch + 1, test_loss_content)
            self.logger.info(
                f'epoch: {epoch + 1}\t| train_loss_content: {train_loss_content:.6f} | train_acc_content: {train_acc_content:.6f} | '
                f'train_acc_domain: {train_acc_domain:.6f} | '
                f'test_loss_content: {test_loss_content:.6f} | test_acc_content: {test_acc_content:.6f} | '
                f'test_acc_domain: {test_acc_domain:.6f}{" | saving best model..." if best else ""}')
            self.metric_writer.add_scalar("test_acc", test_acc_content, self.epoch)
            self.epoch += 1

            if 'confusion' in self.info:
                if self.info['confusion'].get('train', False):
                    self.__plot_confusion_matrix(photo_path=self.confusion_path / f'train-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png',
                                                 labels=train_labels, predicts=train_predicts, classes=list(range(self.num_classes)), normalize=True)

                if self.info['confusion'].get('test', False):
                    self.__plot_confusion_matrix(photo_path=self.confusion_path / f'test-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png',
                                                 labels=test_labels, predicts=test_predicts, classes=list(range(self.num_classes)), normalize=True)

    def train_epoch(self, epoch):  # sourcery skip: low-code-quality
        """
        Main training process
        """
        train_loss_content_src = 0
        train_acc_content = 0
        train_acc_domain = 0
        train_num = 0
        predict, label = [], []
        self.model.train()
        self.classifier_content.train()
        self.classifier_domain.train()
        # domain labels
        label_ones = torch.ones(self.batch_size).long().to(self.device)  # src
        label_zeros = torch.zeros(self.batch_size).long().to(self.device)  # tgt

        for batch, data_pack in enumerate(zip(self.dataloader_source, self.dataloader_target)):
            # `index_t` records the index of target data in the whole dataset, for pseudo labeling.
            # We do not use label_t here.
            (data_s, label_s, _), (data_t, _, _) = data_pack
            data_s, data_t, label_s = data_s.to(self.device), data_t.to(self.device), label_s.to(self.device)

            label.append(label_s.cpu().detach().numpy())
            train_num += label_s.shape[0]

            self.__reset_grad()
            feat_s = self.model(data_s)
            feat_t = self.model(data_t)

            """1. train content classifier on source domain"""
            output_content_src = self.classifier_content(feat_s)
            loss_content_src = self.criterion(output_content_src, label_s)
            predict.append(output_content_src.argmax(dim=1).cpu().detach().numpy())
            train_acc_content += (output_content_src.argmax(dim=1) == label_s).sum().item()

            """2. train domain classifier on both domain"""
            feat_s, feat_t = self.grl(feat_s), self.grl(feat_t)  # GRL
            output_domain_src = self.classifier_domain(feat_s)
            output_domain_tgt = self.classifier_domain(feat_t)
            loss_domain = 0.4 * (self.criterion(output_domain_src, label_ones) +
                                 self.criterion(output_domain_tgt, label_zeros))
            train_acc_domain += (output_domain_src.argmax(dim=1) == label_ones).sum().item() + \
                (output_domain_tgt.argmax(dim=1) == label_zeros).sum().item()

            loss = loss_content_src + loss_domain
            loss.backward()
            self.opt.step()
            self.opt_content.step()
            self.opt_domain.step()

            train_loss_content_src += loss_content_src.item()

            self.lr_scheduler.step()
            self.lr_scheduler_content.step()
            self.lr_scheduler_domain.step()

            if batch % self.train_display == 0:
                print('training... batch: {}/{} loss_content_src: {:.6f} loss_domain: {:.6f}'.format(batch, self.num_train_batch, loss_content_src.item(), loss_domain.item()) +
                      '\b' * len('training... batch: {}/{} loss_content_src: {:.6f} loss_domain: {:.6f}'.format(batch, self.num_train_batch, loss_content_src.item(), loss_domain.item())), end='', flush=True)

        predict = np.concatenate(predict, axis=0)
        label = np.concatenate(label, axis=0)
        return train_loss_content_src / self.num_train_batch, train_acc_content / train_num, train_acc_domain / train_num / 2, label, predict

    def test_epoch(self, epoch):
        # on target domain
        test_loss_content = 0
        test_num = 0
        test_acc_content, test_acc_domain = 0, 0
        label_zeros = torch.zeros(self.batch_size).long().to(self.device)  # tgt

        predict, label = [], []
        self.model.eval()
        self.classifier_content.eval()
        self.classifier_domain.eval()
        with torch.no_grad():
            for data, targets, _ in self.dataloader_test:
                data, targets = data.to(self.device), targets.to(self.device)
                label.append(targets.cpu().detach().numpy())
                test_num += data.shape[0]

                feat = self.model(data)
                output_dontent = self.classifier_content(feat)
                output_domain = self.classifier_domain(feat)
                test_loss_content += self.criterion(output_dontent, targets)

                predict.append(torch.argmax(output_dontent+output_domain, dim=1).cpu().detach().numpy())
                test_acc_content += torch.sum(torch.argmax(output_dontent, dim=1) == targets).item()
                test_acc_domain += torch.sum(torch.argmax(output_domain, dim=1) == label_zeros).item()

        predict = np.concatenate(predict, axis=0)
        label = np.concatenate(label, axis=0)
        return test_loss_content / self.num_test_batch, test_acc_content / test_num, test_acc_domain / test_num, label, predict

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

    def __get_object(self, module, s: str, parameter: dict):
        return getattr(module, s)(**parameter)

    def __get_logger(self):
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.info(f'model: {type(self.model).__name__}')

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
