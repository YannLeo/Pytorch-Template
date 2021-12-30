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
import shutil
import numpy as np

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

warnings.filterwarnings('ignore')

def make_dir(info, config):
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
    shutil.copy(config, save_dir / info['name'] / time_string / 'log')
    return save_dir / info['name'] / time_string


def main(args):
    config, resume = args.config, args.resume
    with open(config) as f:
        info = json.load(f)
    path = make_dir(info, config)
    os.environ['CUDA_VISIBLE_DEVICES'] = info['device']
    tr = getattr(trainer, info['trainer'])(info, resume, path)
    tr.train()
    # train('/home/yl/CodeAndData/data/stft_data_2021_12_3_20_17_224_224.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pytorch Template')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    main(parser.parse_args())

    