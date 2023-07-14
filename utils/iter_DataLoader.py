# -*- encoding: utf-8 -*-
'''
@File        :   iter_DataLoader.py
@Description :   DataLoader which does not raise StopIteration exception
@Time        :   2023/07/11 03:53:59
@Author      :   Leo Yann
@Version     :   1.0
@Contact     :   yangliu991022@gmail.com
'''


class IterDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter = iter(dataloader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataloader)