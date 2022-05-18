import pytest
from mnist_data import MnistDataset
from unittest.mock import Mock
import numpy as np


class TestMnistDataset:

    @pytest.fixture(scope='class')
    def mnist_dataset(self):
        return MnistDataset(batch_size=100, data_path='testdata', download=True)

    def test_data_loaders(self, mnist_dataset):
        assert mnist_dataset.train_loader.batch_size == 100
        assert mnist_dataset.test_loader.batch_size == 100
        # since we mask the data, and the mask are made beforehand
        # based on a fixed batch size, you need ech batch to be the same size
        for batch in iter(mnist_dataset.test_loader):
            x, y = batch
            assert len(x) == 100

    def test_mnist_properties(self, mnist_dataset):
        mnist_iter = iter(mnist_dataset.train_loader)
        example_data, example_targets = next(mnist_iter)
        assert example_data.shape == (100, 1, 28, 28)
        assert example_data.min() == -1
        assert example_data.max() == 1

    @pytest.mark.parametrize('random_offset', [0, 10, 99, 700])
    def test_sample_each_digit(self, mnist_dataset, random_offset):
        np.random.randint = Mock()
        np.random.randint.side_effect = [random_offset]
        sample_x, sample_y = mnist_dataset.sample_each_digit()
        assert sample_y.tolist() == [i for i in range(10)]
        assert sample_x.min() == -1
        assert sample_x.max() == 1





