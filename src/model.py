import torch
from wavenet import Wavenet
from dataclasses import dataclass, field
from model_utils import LossFunctions, ModelRole, LossCalculator
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import json
from pathlib import Path
torch.manual_seed(0)



@dataclass
class Model(ABC):
    input_dim: int
    layer_dim: int
    num_layers: int
    learning_rate: float
    generator: torch.nn.Module = field(init=False)
    generator_optimizer: torch.optim = field(init=False)
    early_stopping_counter = 0
    model_name: str

    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def build_generator(self) -> (torch.nn.Module, torch.optim):
        pass

    @abstractmethod
    def train(self, train_data_loader: DataLoader, test_data_loader: DataLoader, epochs: int, patience: int):
        pass

    @abstractmethod
    def save_model(self, epoch):
        pass

    def save_loss_stats(self, loss_stats: dict):
        with open(Path('models', f'{self.model_name}_loss_stats.json'), 'w') as fp:
            json.dump(loss_stats, fp)

    def get_receptive_field(self, batch_size, sequence_length) -> int:
        # make sure that batch norm is turned off
        self.generator.eval()
        self.generator_optimizer.zero_grad()
        x = np.ones([batch_size, sequence_length, self.input_dim])
        x = Variable(torch.from_numpy(x).float(), requires_grad=True)
        pars = self.generator(x)
        mu = pars[0]
        grad = torch.zeros(mu.size())
        # imagine the last values in the sequence have a gradient
        grad[:, -1, :] = 1.0
        mu.backward(gradient=grad)
        # see what this gradient is wrt the input
        # check for any dimension, how many inputs have a non-zero gradient
        rf = (x.grad.data[0][:, 0] != 0).sum()
        return rf

    def early_stopping(self, best_loss_so_far, new_loss, patience):
        if best_loss_so_far <= new_loss:
            self.early_stopping_counter += 1
        if self.early_stopping_counter > patience:
            return True
        return False


class WN(Model):

    def __post_init__(self):
        self.generator, self.generator_optimizer = self.build_generator()

    def build_generator(self):
        generator = Wavenet(input_dim=self.input_dim, embed_dim=self.layer_dim,
                            loss_function=LossFunctions.MSE, output_dim=self.input_dim, num_layers=self.num_layers,
                            model_type=ModelRole.GENERATOR)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.learning_rate, eps=1e-5)
        return generator, generator_optimizer

    def save_model(self, epoch):
        state = {
                'epoch': epoch,
                'generator': self.generator.state_dict(),
                'gen_optimizer': self.generator_optimizer.state_dict(),
                }
        torch.save(state, Path('models', f'{self.model_name}.pth.tar'))

    def load_model(self):
        state = torch.load(Path(f'models, {self.model_name}.pth.tar'), map_location='cpu')
        self.generator.load_state_dict(state['generator'])

    def train(self, train_data_loader: DataLoader, test_data_loader: DataLoader, epochs: int,
              patience: int):

        sequence_length = self.infer_sequence_length(test_data_loader)
        receptive_field = self.get_receptive_field(train_data_loader.batch_size, sequence_length)
        print(f"The receptive field is {receptive_field}")

        train_mask = self.create_mask(sequence_length, train_data_loader.batch_size, receptive_field)
        test_mask = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)

        loss_calculator = LossCalculator(LossFunctions.MSE)
        best_loss_so_far = 9999999
        loss_stats = {'train': [], 'test': []}
        for epoch in range(epochs):

            train_loss = self.step_trough_batches(train_data_loader, train_mask, loss_calculator, self.train_step)
            test_loss = self.step_trough_batches(test_data_loader, test_mask, loss_calculator, self.test_step)

            loss_stats['train'].append(train_loss)
            loss_stats['test'].append(test_loss)

            print(f"epoch {epoch} | train : {train_loss} test: {test_loss}")

            if self.early_stopping(best_loss_so_far, test_loss, patience) or epoch == epochs-1:
                print(f" Saving model at epoch {epoch} with loss {best_loss_so_far}")
                self.save_model(epoch)
                self.save_loss_stats(loss_stats)
                return epoch
            elif test_loss < best_loss_so_far:
                best_loss_so_far = test_loss

    def step_trough_batches(self, data_loader, mask, loss_calculator, step_function):
        running_loss = 0
        batch_index = 1
        for batch_index, batch in enumerate(data_loader):
            images, _ = batch
            inputs, targets = self.split_images_into_input_target(images)
            loss = step_function(inputs, targets, mask, loss_calculator)
            running_loss += loss
        return running_loss/batch_index

    def evaluate(self, test_data_loader: DataLoader):
        sequence_length = self.infer_sequence_length(test_data_loader)
        receptive_field = self.get_receptive_field(test_data_loader.batch_size, sequence_length)
        print(f"The receptive field is {receptive_field}")
        test_mask = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)
        loss_calculator = LossCalculator(LossFunctions.MSE)
        loss = self.step_trough_batches(test_data_loader, test_mask, loss_calculator, self.test_step)
        return loss

    def infer_sequence_length(self, test_data_loader):
        test_iterator = iter(test_data_loader)
        sample_x, _ = next(test_iterator)
        assert len(sample_x.shape) == 4  # [batch_size, 1, seq_length, num_features]
        assert self.input_dim == sample_x.shape[3]
        sequence_length = sample_x.shape[2] - 1  # the input length is the number of rows - 1
        return sequence_length

    @staticmethod
    def create_mask(sequence_length, batch_size, receptive_field) -> torch.Tensor:
        '''
        make masks to make sure you only back propagate the loss for outputs
        that received a full receptive field of inputs
        '''
        mask = torch.zeros([batch_size, sequence_length])
        mask[receptive_field:, :] = 1
        return mask

    def train_step(self, inputs, targets, train_mask, loss_calculator) -> float:
        self.generator_optimizer.zero_grad()
        outputs = self.generator(inputs)
        loss = loss_calculator.calculate_loss(targets, outputs)
        loss = loss.mean(-1)
        loss = (loss * train_mask).mean(0)
        loss = loss.mean()
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def test_step(self, inputs, targets, test_mask, loss_calculator) -> float:
        self.generator.eval()
        outputs = self.generator(inputs)
        loss = loss_calculator.calculate_loss(targets, outputs)
        loss = loss.sum(-1)
        loss = (loss * test_mask).sum(0)
        loss = loss.mean()
        return loss.item()

    @staticmethod
    def split_images_into_input_target(images) -> (torch.Tensor, torch.Tensor):
        # the last row is not an input because it has not target
        inputs = images[:, 0, :-1, :]
        # the first row is not a target because it has no input
        targets = images[:, 0, 1:, :]
        return inputs, targets
