 # @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 23/10/27 ~ 16:21:22
# @File       : uav.py
# @Note       :

from typing import Literal, Optional
import torch
from torch.utils.data import Dataset
import numpy as np


class UAVDataset(Dataset):
    def __init__(
        self,
        rounds: Optional[list] = None,  # 0~11
        kinds: Optional[list] = None,  # 0~7
        num_samples: int = 0,  # 0~split_ratio*len(data) if train, else remaining
        mode: Literal["train", "test"] = "train",
        split_ratio: float = 0.8,
        path: str = "/home/public/Datasets/UAVData",
    ) -> None:
        super().__init__()
        # skip reading data if no data is needed
        if not rounds or not kinds or num_samples <= 0:
            self.targets = torch.tensor([0])  # dummy; to please dataloader
            print("No data loaded.")
            return

        # start reading data by looping over rounds and kinds
        list_data = []
        list_targets = []
        for round in rounds:
            for label, kind in enumerate(sorted(kinds)):
                # Nx1x2000
                data = np.load(f"{path}/data_{round}/21{kind}.npy")
                # If mode == "train", select the first split_ratio of data, else select the last split_ratio of data, evenly.
                selected = (
                    np.linspace(0, int(split_ratio * len(data)), num=num_samples, endpoint=False, dtype=int)
                    if mode == "train"
                    else np.linspace(int(split_ratio * len(data)), len(data), num=num_samples, endpoint=False, dtype=int)
                )
                data = torch.as_tensor(data[selected], dtype=torch.float32).unsqueeze(1)
                list_data.append(data)
                list_targets.append(torch.full((len(data),), label, dtype=torch.long))

        # end of reading data
        self.data = torch.cat(list_data, dim=0)
        self.targets = torch.cat(list_targets, dim=0)

        print(f"Successfully loaded UAV data -> {self.data.shape}, {self.targets.shape} in rounds {rounds} and kinds {kinds}.")

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data[index]
        # Normalize data
        data = (data - data.mean()) / data.std()
        return data, self.targets[index]

    def __len__(self) -> int:
        return len(self.targets)


class UAVPatchedDataset(UAVDataset):
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data, target = super().__getitem__(index)
        return data.reshape(-1, 200), target


if __name__ == "__main__":
    dataset = UAVDataset(rounds=[0, 11], kinds=[0, 1, 7], num_samples=100, mode="test")
    dataset = UAVDataset(rounds=[0, 11], kinds=[], num_samples=100, mode="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
