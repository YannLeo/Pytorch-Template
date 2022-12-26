# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:18:02
# @File       : _trainer_base.py
# @Note       : The base class of all trainers, and some tools for training

from abc import ABC
import itertools
import logging
import torch
import numpy as np
from pathlib import Path
import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import models
import time

# some tools 


class _Trainer_Base(ABC):
    def __init__(self, info, resume=None, path=Path(), device=torch.device('cuda')):
        self.info = info
        self.resume = resume
        self.device = device
        self.max_epoch = info['epoch']
        self.dataset_train = self.get_object(datasets, info['dataloader_train']['dataset']['type'],
                                             info['dataloader_train']['dataset']['args'])
        self.dataset_val = self.get_object(datasets, info['dataloader_val']['dataset']['type'],
                                           info['dataloader_val']['dataset']['args'])
        self.dataset_test = self.get_object(datasets, info['dataloader_test']['dataset']['type'],
                                            info['dataloader_test']['dataset']['args'])
        self.dataloader_train = torch.utils.data.DataLoader(dataset=self.dataset_train, **info['dataloader_train']['args'])
        self.dataloader_val = torch.utils.data.DataLoader(dataset=self.dataset_val, **info['dataloader_val']['args'])
        self.dataloader_test = torch.utils.data.DataLoader(dataset=self.dataset_test, **info['dataloader_test']['args'])

        self.criterion = self.get_object(torch.nn, info['criterion'], {})
        self.model = self.get_object(models, info['model']['type'], info['model']['args'])
        self.model = self.model.to(self.device)
        self.optimizer = self.get_object(torch.optim, info['optimizer']['type'],
                                         {'params': self.model.parameters(), **info['optimizer']['args']})
        self.lr_scheduler = self.get_object(torch.optim.lr_scheduler, info['lr_scheduler']['type'],
                                            {'optimizer': self.optimizer, **info['lr_scheduler']['args']})
        self.log_path = path / 'log' / 'log.txt'
        self.model_path = path / 'model'
        self.confusion_path = path / 'confusion'
        self.save_period = info['save_period']
        self.min_valid_loss = np.inf
        if self.resume:
            checkpoint = torch.load(self.resume)
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
            self.epoch = checkpoint['epoch'] + 1
        else:
            self.epoch = 0
        self.get_logger()
        self.logger.info(f'model: {type(self.model).__name__}')

    def get_object(self, module, s: str, parameter: dict):
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
            print(f'epoch: {epoch + 1}\t| ', end='')
            train_loss, train_acc, train_labels, train_predicts = self.train_epoch(epoch + 1)
            print('train_loss: {:.6f} | train_acc: {:.6f} | '.format(train_loss, train_acc), end='')

            print('validating...' + '\b' * len('validating...'), end='', flush=True)
            valid_loss, valid_acc, valid_labels, valid_predicts = self.valid_epoch(epoch + 1)
            time_end = time.time()
            print('valid_loss: {:.6f} | valid_acc: {:.6f} | time: {:d}s | '.format(valid_loss, valid_acc,
                                                                                   int(time_end - time_begin)), end='')

            print('tesing...' + '\b' * len('tesing...'), end='', flush=True)
            test_loss, test_acc, *_ = self.test_epoch(epoch + 1)
            print('test_loss: {:.6f} | test_acc: {:.6f}'.format(test_loss, test_acc), end='')
            best = self.save_model_by_valid_loss(epoch + 1, valid_loss)

            self.lr_scheduler.step()
            self.logger.info(
                'epoch: {} \t| train_loss: {:.6f} | train_acc: {:.6f} | valid_loss: {:.6f} | valid_acc: {:.6f} | test_acc: {:.6f} '.format(
                    epoch + 1, train_loss, train_acc, valid_loss, valid_acc, test_acc)
                + (' | saving best model...' if best else ''))
            self.epoch += 1

            if 'confusion' in self.info:
                if 'train' in self.info['confusion'] and self.info['confusion']['train']:
                    self.plot_confusion_matrix(photo_path=(self.confusion_path / f'train-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png'),
                                               labels=train_labels, predicts=train_predicts, classes=list(range(self.info['model']['args']['num_class'])), normalize=True)

                if 'test' in self.info['confusion'] and self.info['confusion']['test']:
                    self.plot_confusion_matrix(photo_path=(self.confusion_path / f'test-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png'),
                                               labels=valid_labels, predicts=valid_predicts, classes=list(range(self.info['model']['args']['num_class'])), normalize=True)

            else:
                self.plot_confusion_matrix(photo_path=(self.confusion_path / f'test-{str(epoch + 1).zfill(len(str(self.max_epoch)))}.png'),
                                           labels=valid_labels, predicts=valid_predicts, classes=list(range(self.info['model']['args']['num_class'])), normalize=True)
            # plot_cm(self.model, self.confusion_path / '{}.png'.format(epoch + 1), self.dataloader_test, list(range(self.info['model']['args']['num_class'])), self.device)

    def train_epoch(self, epoch):
        train_loss = 0
        train_acc_num = 0
        predict, label = [], []
        self.model.train()
        for batch, (data, target) in enumerate(self.dataloader_train):
            data, target = data.to(self.device), target.to(self.device)
            out = self.model(data)
            loss = self.criterion(out, target)
            predict.append(torch.argmax(out, dim=1).cpu().detach().numpy())
            label.append(target.cpu().detach().numpy())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc_num += torch.sum(torch.argmax(out, dim=1) == target).item()
            if batch % self.train_display == 0:
                print('training... batch: {}/{} Loss: {:.6f}'.format(batch, self.num_train_batch, loss.item()) +
                      '\b' * len(
                    'training... batch: {}/{} Loss: {:.6f}'.format(batch, self.num_train_batch, loss.item())), end='',
                    flush=True)
        predict = np.concatenate(predict, axis=0)
        label = np.concatenate(label, axis=0)
        return train_loss / self.num_train_batch, train_acc_num / self.num_train, label, predict

    def valid_epoch(self, epoch):  # sourcery skip: remove-unused-enumerate
        test_loss = 0
        test_acc_num = 0
        predict, label = [], []
        self.model.eval()
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.dataloader_val):
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                predict.append(torch.argmax(out, dim=1).cpu().detach().numpy())
                label.append(target.cpu().detach().numpy())
                loss = self.criterion(out, target)
                test_loss += loss.item()
                test_acc_num += torch.sum(torch.argmax(out, dim=1) == target).item()
        predict = np.concatenate(predict, axis=0)
        label = np.concatenate(label, axis=0)
        return test_loss / self.num_test_batch, test_acc_num / self.num_test, label, predict

    def test_epoch(self, epoch):  # sourcery skip: remove-unused-enumerate
        test_loss = 0
        test_acc_num = 0
        predict, label = [], []
        self.model.eval()
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.dataloader_test):
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                predict.append(torch.argmax(out, dim=1).cpu().detach().numpy())
                label.append(target.cpu().detach().numpy())
                loss = self.criterion(out, target)
                test_loss += loss.item()
                test_acc_num += torch.sum(torch.argmax(out, dim=1) == target).item()
        predict = np.concatenate(predict, axis=0)
        label = np.concatenate(label, axis=0)
        return test_loss / self.num_test_batch, test_acc_num / self.num_test, label, predict

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
        elif epoch % self.save_period == 0:
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
            filename = str(self.model_path / f'checkpoint-epoch{epoch}.pth')
            torch.save(state, filename)

    def resume_checkpoint(self):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(self.resume)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.min_valid_loss = checkpoint['loss_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

    def plot_confusion_matrix(self, photo_path, labels, predicts, classes, normalize=False, title='Confusion matrix',
                              cmap=plt.cm.Oranges):
        FONT_SIZE = 8
        cm = confusion_matrix(labels, predicts, labels=list(range(len(classes))))
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            # print(cm)
        plt.figure(figsize=(8 * 2, 6 * 2))
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
                     fontsize=FONT_SIZE,
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(photo_path)

    def get_logger(self):
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def plot_label_clusters(self, loader1, loader2, ratio=1.0):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        all_feats, all_targets = [], []
        num_valid = int(loader1.batch_size * ratio)  # 防止数据量过大, 每个 batch 只取一部分数据. loader 是否 shuffle 问题都不大

        self.model.eval()
        with torch.no_grad():
            for data1, targets1 in loader1:
                feat1 = self.model.feature_extractor(data1.to(self.device)[:num_valid])
                all_feats.append(feat1.cpu().numpy())
                all_targets.append(targets1.numpy()[:num_valid])

            z = np.concatenate(all_feats, axis=0)
            labels = np.concatenate(all_targets, axis=0)
            z = TSNE(2, init='pca', learning_rate='auto', verbose=1).fit_transform(z)

            plt.figure(figsize=(16, 14))
            plt.scatter(z[:, 0], z[:, 1], c=labels)
            plt.colorbar()
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            # plt.xlim(-150, 150)
            # plt.ylim(-150, 150)

            plt.savefig(self.confusion_path / "loader1_clusters.png")
            # plt.show()

            if loader2 is not None:
                labels_domain = [np.zeros(labels.shape[0])]
                for data2, _ in loader2:
                    feat2 = self.model.feature_extractor(data2.to(self.device)[:num_valid])
                    all_feats.append(feat2.cpu().numpy())

                z = np.concatenate(all_feats, axis=0)
                z = TSNE(2, init='pca', learning_rate='auto', verbose=1).fit_transform(z)
                labels_domain.append(np.ones(z.shape[0] - labels.shape[0]))
                labels_domain = np.concatenate(labels_domain, axis=0)

                plt.figure(figsize=(16, 14))
                plt.scatter(z[:, 0], z[:, 1], c=labels_domain)
                plt.colorbar()
                plt.xlabel("z[0]")
                plt.ylabel("z[1]")
                # plt.xlim(-150, 150)
                # plt.ylim(-150, 150)
                plt.savefig(self.confusion_path / "domain_clusters.png")
                # plt.show()
