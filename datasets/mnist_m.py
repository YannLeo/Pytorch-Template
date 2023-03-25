import pickle as pkl
from torch.utils.data import Dataset
import torch
from torchvision import transforms


class MNIST_MDataset(Dataset):
    def __init__(self, path="/home/public/Datasets/MNIST-M/mnist_m_data.pkl", train=True):
        super().__init__()
        with open(path, "rb") as f:
            data_dict = pkl.load(f)

        if train:
            self.data, self.targets = data_dict["train"], data_dict["train_label"]  # (60000, 28, 28, 3)
        else:
            self.data, self.targets = data_dict["val"], data_dict["val_label"]  # (10000, 28, 28, 3)

        print()
        self.targets = torch.from_numpy(self.targets).long()

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # automatically normalize to [0, 1]
            # transforms.Normalize([0.4635, 0.4673, 0.4198], [0.2546, 0.2380, 0.2617]),
        ])
        print("Successfully loaded MNIST_M data ->", self.data.shape, self.targets.shape)

    def __getitem__(self, index):
        return self.transform(self.data[index]).float(), self.targets[index]

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    import torch

    def get_mean_std_value(loader):

        data_sum, data_squared_sum, num_batches = 0, 0, 0

        for data, _ in loader:
            # data: [batch_size,channels,height,width]
            # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
            data_sum += torch.mean(data, dim=[0, 2, 3])    # [batch_size,channels,height,width]
            # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
            data_squared_sum += torch.mean(data**2, dim=[0, 2, 3])  # [batch_size,channels,height,width]
            # 统计batch的数量
            num_batches += 1
        # 计算均值
        mean = data_sum/num_batches
        # 计算标准差
        std = (data_squared_sum/num_batches - mean**2)**0.5
        return mean, std

    train_loader = torch.utils.data.DataLoader(
        MNIST_MDataset("/home/rxy/Datasets/MNIST-M/mnist_m_data.pkl", train=True), batch_size=1000, shuffle=False)
    mean, std = get_mean_std_value(train_loader)
    print(f'mean = {mean}, std = {std}')
