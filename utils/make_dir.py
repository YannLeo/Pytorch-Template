import os
from pathlib import Path
import time
import shutil

def make_dir(info, config):
    save_dir = Path('saved')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir / info['name']):
        os.mkdir(save_dir / info['name'])
    time_list = list(time.localtime(time.time()))
    time_string = '-'.join(map(lambda x: str(x).zfill(2), time_list[:6]))
    if not os.path.exists(save_dir / info['name'] / time_string):
        os.mkdir(save_dir / info['name'] / time_string)
        os.mkdir(save_dir / info['name'] / time_string / 'log')
        os.mkdir(save_dir / info['name'] / time_string / 'model')
        os.mkdir(save_dir / info['name'] / time_string / 'confusion_matrix')
    shutil.copy(config, save_dir / info['name'] / time_string / 'log')
    return save_dir / info['name'] / time_string
