import numpy as np

from models import WN
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
    def mock_data_loader(self):
        mds = MockDataSet(input_dim=self.input_dim, sequence_length=self.sequence_length+1,
                          number_of_examples=self.number_of_examples)
        return DataLoader(mds, batch_size=self.batch_size)

    @pytest.fixture(scope='class')
    def wn_4_layers(self):
        return WN(input_dim=self.input_dim, layer_dim=128, num_layers=4, learning_rate=0.05,
                  model_name='test_model')

    @pytest.fixture(scope='class')
    def wn_2_layers(self):
        return WN(input_dim=self.input_dim, layer_dim=128, num_layers=2, learning_rate=0.05,
                  model_name='test_model')

    def test_receptive_field(self, wn_4_layers, wn_2_layers):
        '''
        the receptive field = 2^(num_layers -1) * kernel_size
        '''
        # can choose any sequence_length, as long as more then receptive field
        rf = wn_4_layers.get_receptive_field()
        assert rf == 2**(4-1) * 2

        rf = wn_2_layers.get_receptive_field()
        assert rf == 2**(2-1) * 2

    def test_early_stopping(self, wn_2_layers, mock_data_loader):
        wn_2_layers.save_model = Mock()  # make sure the model doesnt actually gets saved
        wn_2_layers.save_loss_stats = Mock()
        wn_2_layers.step_trough_batches = Mock()
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
        wn_2_layers.step_trough_batches.side_effect = \
            [{'mse': mse_loss} for mse_loss in [1, 5,
                                                1, 4,
                                                1, 3,
                                                1, 2,
                                                1, 1,
                                                1, 2,  # patience counter = 1
                                                1, 1,
                                                1, 2,  # patience counter = 2
                                                1, 2,  # patience counter = 3 -> aborts training
                                                1, 2,
                                                1, 2]]

        epoch = wn_2_layers.train(mock_data_loader, mock_data_loader, epochs=100, patience=3)
        assert epoch == 8

    @staticmethod
    def _get_params_from_model(model) -> np.array:
        return [p for p in model.generator.parameters()][0].clone().detach().numpy()

    def test_test_step(self, wn_2_layers, mock_data_loader):
        initial_parameters = self._get_params_from_model(wn_2_layers)
        mask = wn_2_layers.create_mask(self.sequence_length, self.batch_size, receptive_field=2**(2-1) * 2)
        sample = next(iter(mock_data_loader))
        image, _ = sample
        x, y = wn_2_layers.split_images_into_input_target(image)
        wn_2_layers.test_step(x, y, mask)
        parameters_after_test_step = self._get_params_from_model(wn_2_layers)
        assert (initial_parameters == parameters_after_test_step).all()

    def test_train_step(self, wn_2_layers, mock_data_loader):
        initial_parameters = self._get_params_from_model(wn_2_layers)
        mask = wn_2_layers.create_mask(self.sequence_length, self.batch_size, receptive_field=2**(2-1) * 2)
        sample = next(iter(mock_data_loader))
        image, _ = sample
        x, y = wn_2_layers.split_images_into_input_target(image)
        wn_2_layers.train_step(x, y, mask)
        parameters_after_train_step = self._get_params_from_model(wn_2_layers)
        assert not (initial_parameters == parameters_after_train_step).all()

    def test_freerunning(self, wn_2_layers):
        receptive_field=2**(2-1) * 2
        wn_2_layers.generator = Mock()
        batch_size = 10
        x = torch.ones((batch_size, self.sequence_length, self.input_dim))
        wn_2_layers.generator.side_effect = [[torch.zeros_like(x)] for _ in range(self.sequence_length - receptive_field)]
        result = wn_2_layers.free_running(x, receptive_field)
        assert result.sum() == receptive_field * self.input_dim * batch_size

    def test_create_mask(self, wn_2_layers):
        mask = wn_2_layers.create_mask(sequence_length=88, batch_size=99, receptive_field=14)
        assert mask.size() == (99, 88)
        assert mask.sum() == (88-14) * 99


class TestSWNModel:

    input_dim = 4
    sequence_length = 88
    number_of_examples = 100
    batch_size = 64

    @pytest.fixture(scope='class')
    def swn_4_layers(self):
        return WN(input_dim=self.input_dim, layer_dim=128, num_layers=4, learning_rate=0.05,
                  model_name='test_model')

    @pytest.fixture(scope='class')
    def swn_2_layers(self):
        return WN(input_dim=self.input_dim, layer_dim=128, num_layers=2, learning_rate=0.05,
                  model_name='test_model')

    @pytest.fixture(scope='class')
    def mock_data_loader(self):
        mds = MockDataSet(input_dim=self.input_dim, sequence_length=self.sequence_length+1,
                          number_of_examples=self.number_of_examples)
        return DataLoader(mds, batch_size=self.batch_size)

    def test_receptive_field(self, swn_4_layers, swn_2_layers):
        '''
        the receptive field = 2^(num_layers -1) * kernel_size
        '''
        # can choose any sequence_length, as long as more then receptive field
        rf = swn_4_layers.get_receptive_field()
        assert rf == 2**(4-1) * 2

        rf = swn_2_layers.get_receptive_field()
        assert rf == 2**(2-1) * 2

    def test_test_step(self, swn_2_layers, mock_data_loader):
        initial_parameters = [p for p in swn_2_layers.generator.parameters()][0].clone().detach().numpy()
        mask = swn_2_layers.create_mask(self.sequence_length, self.batch_size, receptive_field=2**(2-1) * 2)
        sample = next(iter(mock_data_loader))
        image, _ = sample
        x, y = swn_2_layers.split_images_into_input_target(image)
        swn_2_layers.test_step(x, y, mask)
        parameters_after_test_step = [p for p in swn_2_layers.generator.parameters()][0].clone().detach().numpy()
        assert (initial_parameters == parameters_after_test_step).all()
        assert not swn_2_layers.generator.training

    def test_generation_no_targets(self, swn_2_layers, mock_data_loader):
        sample = next(iter(mock_data_loader))
        image, _ = sample
        x, y = swn_2_layers.split_images_into_input_target(image)
        swn_2_layers.generator(x)
        assert not swn_2_layers.generator.training

    def test_train_step(self, swn_2_layers, mock_data_loader):
        initial_parameters = [p for p in swn_2_layers.generator.parameters()][0].clone().detach().numpy()
        mask = swn_2_layers.create_mask(self.sequence_length, self.batch_size, receptive_field=2**(2-1) * 2)
        sample = next(iter(mock_data_loader))
        image, _ = sample
        x, y = swn_2_layers.split_images_into_input_target(image)
        swn_2_layers.train_step(x, y, mask)
        parameters_after_train_step = [p for p in swn_2_layers.generator.parameters()][0].clone().detach().numpy()
        assert not (initial_parameters == parameters_after_train_step).all()
        assert swn_2_layers.generator.training


