import numpy as np
import torch
from torch.utils.data import Dataset


#Given the path of x and path of y, return a datatset that contains the tensor x and tensor y
#(Str, Str) -> (Dataset)
class GalaxyData(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode = 'r+')
        self.x = torch.from_numpy(self.x).type(torch.float32)

        self.y = np.load(y_path, mmap_mode= 'r+')
        self.y = torch.from_numpy(self.y).type(torch.long)

        self.length = self.y.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.length