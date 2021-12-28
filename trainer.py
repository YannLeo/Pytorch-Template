import torch
import 


class BaseTrainer():
    def __init__(self, info, resume=None, device=torch.device('cuda:4')):
        self.info = info
        self.resume = resume
        self.device = device
        self.dataset_train = self.get_object()
        

    def get_object(self, s:str, module, parameter):
        pass

    
    def train(self):
        pass

    def train_epoch(self, epoch)