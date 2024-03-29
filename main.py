import os
import argparse
import warnings

import toml
import torch
import numpy as np

from utils.make_dir import make_dir
from utils import clean_at_exit


"""0. Setting the random seed"""
SEED = 2023
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # False
np.random.seed(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


"""1. Reading configs from console"""
parser = argparse.ArgumentParser("Pytorch Template")
parser.add_argument("-c", "--config", default="configs/uav.toml", type=str, help="config file path (default: None)")
parser.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None).")


"""2. Loading toml configuration file"""
config, resume = parser.parse_args().config, parser.parse_args().resume
with open(config, "r", encoding="utf8") as f:
    info = toml.load(f)


"""3. Start training ..."""
path = make_dir(info, config, resume)
# register do_clean when `Ctrl+C` is pressed; maybe there is no object named `trainer` when `Ctrl+C` is pressed
train_end = lambda: locals()["trainer"]._train_end() if locals().get("trainer") else None
clean_at_exit.clean_after_killed(path, train_end)

print(f"--- Using configuration file: {config} ---")
print(f"--- Using device(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')} ---")

# load trainer from toml file
import trainers  # move here to filter unnecessary warnings

try:
    trainer: trainers.TrainerBase = getattr(trainers, info["trainer"])(info, path)
    trainer.train()
except Exception as e:
    import traceback
    print(f"Error occurred and here is the traceback ↓↓↓\n======\n{traceback.format_exc()}======")
    clean_at_exit.do_clean(path, train_end)
