from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataclasses import dataclass, field


@dataclass
class MnistDataset:

    batch_size: int
    train_loader: DataLoader = field(init=False)
    test_loader: DataLoader = field(init=False)

    def __post_init__(self):
        self.train_loader, self.test_loader = self.prepare_data_loaders()

    def prepare_data_loaders(self) -> (DataLoader, DataLoader):
        train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
        test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(), download=True)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=1,
                                  drop_last=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, num_workers=1,
                                 drop_last=True)
        return train_loader, test_loader
