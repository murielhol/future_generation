from model import WN

from dataclasses import dataclass
import pytest
from torch.utils.data import DataLoader, Dataset
from unittest.mock import Mock
import torch


@dataclass
class MockDataSet(Dataset):
    number_of_examples: int
    input_dim: int
    sequence_length: int

    def __len__(self):
        return self.number_of_examples

    def __getitem__(self, idx):
        x = torch.rand(1, self.sequence_length, self.input_dim)
        y = torch.randint(low=0, high=9, size=(1,))
        return x, y


class TestWNModel:

    input_dim = 4
    sequence_length = 88
    number_of_examples = 100
    batch_size = 64

    @pytest.fixture(scope='class')
    def mock_data_loaders(self):
        mds = MockDataSet(input_dim=self.input_dim, sequence_length=self.sequence_length,
                          number_of_examples=self.number_of_examples)
        return DataLoader(mds, batch_size=self.batch_size)

    @pytest.fixture(scope='class')
    def wn_4_layers(self):
        return WN(input_dim=self.input_dim, layer_dim=128, num_layers=4, learning_rate=0.005)

    @pytest.fixture(scope='class')
    def wn_2_layers(self):
        return WN(input_dim=self.input_dim, layer_dim=128, num_layers=2, learning_rate=0.005)

    def test_receptive_field(self, wn_4_layers, wn_2_layers, mock_data_loaders):
        '''
        the receptive field = 2^(num_layers -1) * kernel_size
        '''
        # can choose any sequence_length, as long as more then receptive field
        sequence_length = 100
        rf = wn_4_layers.get_receptive_field(mock_data_loaders.batch_size, sequence_length)
        assert rf == 2**(4-1) * 2

        rf = wn_2_layers.get_receptive_field(mock_data_loaders.batch_size, sequence_length)
        assert rf == 2**(2-1) * 2

    def test_early_stopping(self, wn_2_layers, mock_data_loaders):
        wn_2_layers.save_model = Mock()  # make sure the model doesnt actually gets saved
        wn_2_layers.apply_batches = Mock()
        '''mock the data such that it will be
        
        epoch 0 | train : 1 test: 5
        epoch 1 | train : 1 test: 4
        epoch 2 | train : 1 test: 3
        epoch 3 | train : 1 test: 2
        epoch 4 | train : 1 test: 1
        epoch 5 | train : 1 test: 2
        epoch 6 | train : 1 test: 1
        epoch 7 | train : 1 test: 2
        epoch 8 | train : 1 test: 2
        '''
        wn_2_layers.apply_batches.side_effect = [1, 5,
                                                 1, 4,
                                                 1, 3,
                                                 1, 2,
                                                 1, 1,
                                                 1, 2,  # patience counter = 1
                                                 1, 1,
                                                 1, 2,  # patience counter = 2
                                                 1, 2,  # patience counter = 3 -> aborts training
                                                 1, 2,
                                                 1, 2]
        epoch = wn_2_layers.train(mock_data_loaders, mock_data_loaders, epochs=100, patience=3)
        assert epoch == 8



