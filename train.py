import os
import json
import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import trainer
import warnings

warnings.filterwarning('ignore')

def train(path_h5):
    EPOCH = 50
    batch_size = 32
    device = torch.device('cuda:4')
    model = ResNet50(num_class=8)
    model = model.to(device)
    critern = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
    dataset_train = H5Dataset(path=path_h5, train=True)
    dataset_test = H5Dataset(path=path_h5, train=False)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    num_train = len(dataset_train)
    num_test = len(dataset_test)
    num_train_batch = num_train // batch_size
    num_test_batch = num_test // batch_size
    
    for epoch in range(EPOCH):
        print('epoch: {} \t| '.format(epoch), end='')
        train_loss = 0
        train_acc_num = 0
        model.train()
        for i, (data, target) in enumerate(dataloader_train):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = critern(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc_num += torch.sum(torch.argmax(out, dim=1) == target).item()
        print('train_loss: {:.6f} | train_acc: {:.6f} | '.format(train_loss / num_train_batch, train_acc_num / num_train), end='')
        test_loss = 0
        test_acc_num = 0
        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader_test):
                data, target = data.to(device), target.to(device)
                out = model(data)
                loss = critern(out, target)
                test_loss += loss.item()
                test_acc_num += torch.sum(torch.argmax(out, dim=1) == target).item()
        print('test_loss: {:.6f} | test_acc: {:.6f}'.format(test_loss / num_test_batch, test_acc_num / num_test))
        lr_scheduler.step()

def make_dir(info):
    save_dir = Path('saved')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir / info['name']):
        os.mkdir(save_dir / info['name'])
    time_list = list(time.localtime(time.time()))
    time_string = str(time_list[0]) + '-' + str(time_list[1]) + '-' + str(time_list[2]) + '-' + str(time_list[3]) + '-' + str(time_list[4]) + '-' + str(time_list[5])
    if not os.path.exists(save_dir / info['name'] / time_string):
        os.mkdir(save_dir / info['name'] / time_string)
        os.mkdir(save_dir / info['name'] / time_string / 'log')
        os.mkdir(save_dir / info['name'] / time_string / 'model')
    return save_dir / info['name'] / time_string

    

def main(args):
    config, resume = args.config, args.resume
    with open(config) as f:
        info = json.load(f)
    path = make_dir(info)
    tr = getattr(trainer, info['trainer'])(info, resume, path)
    tr.train()
    train('/home/yl/CodeAndData/data/stft_data_2021_12_3_20_17_224_224.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pytorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    main(parser.parse_args())

    