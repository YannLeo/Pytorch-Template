# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 22/12/26 ~ 16:20:01
# @File       : basic_trainer.py
# @Note       : A basic trainer for training a feed forward neural network 

from _trainer_base import _Trainer_Base
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


class BasicTrainer(_Trainer_Base):
    pass