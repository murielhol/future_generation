import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np


@dataclass
class DataSet(ABC):

    batch_size: int
    train_loader: DataLoader = field(init=False)
    test_loader: DataLoader = field(init=False)

    def __post_init__(self):
        self.train_loader, self.test_loader = self.prepare_data_loaders()

    @abstractmethod
    def prepare_data_loaders(self) -> (DataLoader, DataLoader):
        raise NotImplementedError


@dataclass
class MinMaxScaler(object):

    def __call__(self, img):
        return img * 2 - 1


@dataclass
class MnistDataset(DataSet):

    data_path: str
    download: bool

    def prepare_data_loaders(self) -> (DataLoader, DataLoader):
        train_data = datasets.MNIST(root=self.data_path, train=True,
                                    transform=Compose([ToTensor(), MinMaxScaler()]), download=self.download)
        test_data = datasets.MNIST(root=self.data_path, train=False,
                                   transform=Compose([ToTensor(), MinMaxScaler()]), download=self.download)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                 drop_last=True)
        return train_loader, test_loader

    def sample_each_digit(self):
        test_data = datasets.MNIST(root=self.data_path, train=False, transform=ToTensor(), download=self.download)
        number_of_datapoints = test_data.targets.shape[0]
        random_offset = np.random.randint(low=0, high=int(number_of_datapoints/10-1))
        # check how many examples per digit, so you know where in the sorted data to look
        labels_counts = [v for (_, v) in sorted(Counter(test_data.targets.tolist()).items())]
        indexes = [random_offset] + [sum(labels_counts[:i]) + random_offset for i in range(1, 10)]
        sorted_index = torch.argsort(test_data.targets)
        # note that we need to do the normalization ourselves here
        images = (test_data.data[sorted_index[indexes]] - 127.5) / 127.5
        return images, test_data.targets[sorted_index[indexes]]


