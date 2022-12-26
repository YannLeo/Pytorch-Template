import os
import toml
import argparse
import torch
import trainers
import warnings
import numpy as np
from utils.make_dir import make_dir


"""0. Setting the random seed"""
SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

warnings.filterwarnings('ignore')


"""1. Reading configs from console"""
parser = argparse.ArgumentParser('Pytorch Template')
parser.add_argument('-c', '--config', default='configs/147319.toml',  # default
                    type=str, help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')


"""2. Loading toml configuration file"""
config, resume = parser.parse_args().config, parser.parse_args().resume
with open(config, 'r', encoding='utf8') as f:
    info = toml.load(f)


"""3. Start training ..."""
path = make_dir(info, config)
print(f"--- Using configuration file: {config} ---")
print(f"--- Using device(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')} ---")

trainer = getattr(trainers, info['trainer'])(info, resume, path)  # load trainer from toml file
trainer.train()
