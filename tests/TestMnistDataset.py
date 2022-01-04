import pytest
from mnist_data import MnistDataset


class TestMnistDataset:

    @pytest.fixture(scope='class')
    def mnist_dataset(self):
        return MnistDataset(batch_size=100)

    def test_data_loaders(self, mnist_dataset):
        assert mnist_dataset.train_loader.batch_size == 100
        assert mnist_dataset.test_loader.batch_size == 100
        # since we mask the data, and the mask are made beforehand
        # based on a fixed batch size, you need ech batch to be the same size
        for batch in iter(mnist_dataset.test_loader):
            x, y = batch
            assert len(x) == 100

    def test_mnist_properties(self, mnist_dataset):
        mnist_iter = enumerate(mnist_dataset.train_loader)
        batch_idx, (example_data, example_targets) = next(mnist_iter)
        assert example_data.shape == (100, 1, 28, 28)
        assert example_data[0].min() >= 0
        assert example_data[0].max() <= 1





