from torch.utils.data import Dataset
import numpy as np
import torch
import h5py

class H5Dataset(Dataset):
    '''
    path:       path of h5 file
    valid_rate: num of train dataset / num of total dataset
    train:      if true, return train dataset; else, return test dataset
    '''
    def __init__(self, path='stft_data_2021_12_3_20_17_224_224.h5', valid_rate=0.7, train=True):
        super().__init__()
        data, targets = [], []
        kinds = ['1','2','3','4','5','6','7','8']
        # read data from h5 file and split data into train dataset and valid dataset
        with h5py.File(path, 'r') as f:
            for key in kinds:
                temp = f[key]
                num = temp.shape[0]
                if train:
                    data.append(temp[:int(num*valid_rate)])
                else:
                    data.append(temp[int(num*valid_rate):])
                targets.append((int(key)-1) * np.ones(data[-1].shape[0], dtype=np.int64))
                print(key + ' done.')
        self.data = torch.from_numpy(np.concatenate(data, 0))
        self.targets = torch.from_numpy(np.concatenate(targets, 0))
        print(self.data.shape, self.targets.shape)
        assert self.data.shape[0] == self.targets.shape[0]

    def __getitem__(self, index):
        data = self.data[index].float()
        target = self.targets[index].long()

        # normalize
        max_num = data.abs().max()
        data = data / max_num
        
        return data, target

    def __len__(self):
        return self.targets.shape[0]
