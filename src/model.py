import math
from typing import List, Dict

import torch
from wavenet import Wavenet
from swavenet import StochasticWavenet
from dataclasses import dataclass, field
from torch.autograd import Variable
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import json
from pathlib import Path
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


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

    def get_receptive_field(self) -> int:
        sequence_length = 1000
        # make sure that batch norm is turned off
        self.generator.eval()
        self.generator_optimizer.zero_grad()
        x = np.ones([1, sequence_length, self.input_dim])
        x = Variable(torch.from_numpy(x).float(), requires_grad=True)
        mask = self.create_mask(sequence_length, 1, sequence_length-1)
        generated = self.generator(x, x.clone(), mask)[0]
        grad = torch.zeros(generated.size())
        # imagine the last values in the sequence have a gradient
        grad[:, -1, :] = 1.0
        generated.backward(gradient=grad)
        # see what this gradient is wrt the input
        # check for any dimension, how many inputs have a non-zero gradient
        rf = (x.grad.data[0][:, 0] != 0).sum()
        assert sequence_length > rf.item(), f'{sequence_length} is smaller then the receptive field, increase value'
        return rf

    def early_stopping(self, best_loss_so_far, new_loss, patience):
        if best_loss_so_far <= new_loss:
            self.early_stopping_counter += 1
        if self.early_stopping_counter > patience:
            return True
        return False

    @staticmethod
    def split_images_into_input_target(images) -> (torch.Tensor, torch.Tensor):
        # the last row is not an input because it has not target
        inputs = images[:, 0, :-1, :]
        # the first row is not a target because it has no input
        targets = images[:, 0, 1:, :]
        return inputs, targets

    def visualize_performance(self, images: torch.Tensor):
        receptive_field = self.get_receptive_field()
        images_result = self.free_running(images.clone(), receptive_field)
        # reverse black and white for the generated part for visualization
        images[:, receptive_field:, :] = images[:, receptive_field:, :] * -1
        images_result[:, receptive_field:, :] = images_result[:, receptive_field:, :] * -1
        vis = torch.cat([images, images_result], dim=0)[:, None, :, :]
        grid = torchvision.utils.make_grid(vis, nrow=10, pad_value=1)
        plt.imshow(grid.permute(1, 2, 0))
        plt.savefig(Path('models', f'{self.model_name}.png'))

    def infer_sequence_length(self, test_data_loader):
        test_iterator = iter(test_data_loader)
        sample_x, _ = next(test_iterator)
        assert len(sample_x.shape) == 4  # [batch_size, 1, seq_length, num_features]
        assert self.input_dim == sample_x.shape[3]
        sequence_length = sample_x.shape[2] - 1  # the input length is the number of rows - 1
        return sequence_length

    def free_running(self, images, receptive_field):
        sequence_length = images.shape[1]
        mask = self.create_mask(sequence_length, images.shape[0], receptive_field)
        for i in range(sequence_length - receptive_field):
            sample_input = images[:, i:i+receptive_field, :]
            sample_output = self.generator(sample_input, sample_input.clone(), mask)[0]
            # replace the original row with the generated row
            images[:, i+receptive_field, :] = sample_output[:, -1, :]
        return images

    @staticmethod
    def create_mask(sequence_length, batch_size, receptive_field) -> torch.Tensor:
        '''
        make masks to make sure you only back propagate the loss for outputs
        that received a full receptive field of inputs
        '''
        mask = torch.zeros([batch_size, sequence_length])
        mask[receptive_field:, :] = 1
        return mask


class WN(Model):

    def __post_init__(self):
        self.generator, self.generator_optimizer, self.generator_loss_function = self.build_generator()

    def build_generator(self):
        generator = Wavenet(input_dim=self.input_dim, layer_dim=self.layer_dim, output_dim=self.input_dim,
                            num_layers=self.num_layers)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.learning_rate, eps=1e-5)
        generator.eval()
        return generator, generator_optimizer, torch.nn.MSELoss()

    def save_model(self, epoch):
        state = {
                'epoch': epoch,
                'generator': self.generator.state_dict(),
                'gen_optimizer': self.generator_optimizer.state_dict(),
                }
        torch.save(state, Path('models', f'{self.model_name}.pth'))

    def load_model(self):
        state = torch.load(Path('models', f'{self.model_name}.pth'), map_location='cpu')
        self.generator.load_state_dict(state['generator'])
        self.generator_optimizer.load_state_dict(state['gen_optimizer'])

    def train(self, train_data_loader: DataLoader, test_data_loader: DataLoader, epochs: int,
              patience: int):

        receptive_field = self.get_receptive_field()
        print(f"The receptive field is {receptive_field}")

        sequence_length = self.infer_sequence_length(test_data_loader)
        train_mask = self.create_mask(sequence_length, train_data_loader.batch_size, receptive_field)
        test_mask = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)

        loss_calculator = self.generator_loss_function
        best_loss_so_far = 9999999
        loss_stats = {'train_mse': [], 'test_mse': []}

        epoch = 0
        for epoch in range(epochs):

            train_loss = self.step_trough_batches(train_data_loader, train_mask, loss_calculator, self.train_step)
            test_loss = self.step_trough_batches(test_data_loader, test_mask, loss_calculator, self.test_step)

            loss_stats['train_mse'].append(train_loss['mse'])
            loss_stats['test_mse'].append(test_loss['mse'])

            print(f"epoch {epoch} | train : {train_loss} test: {test_loss}")

            if self.early_stopping(best_loss_so_far, test_loss['mse'], patience) or epoch == epochs-1:
                print(f" Saving model at epoch {epoch} with loss {best_loss_so_far}")
                self.save_model(epoch)
                self.save_loss_stats(loss_stats)
                return epoch
            elif test_loss['mse'] < best_loss_so_far:
                best_loss_so_far = test_loss['mse']
        self.save_model(epoch)

    def evaluate(self, test_data_loader: DataLoader):
        self.load_model()
        receptive_field = self.get_receptive_field()
        print(f"The receptive field is {receptive_field}")
        sequence_length = self.infer_sequence_length(test_data_loader)
        test_mask = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)
        loss_calculator = self.generator_loss_function
        loss = self.step_trough_batches(test_data_loader, test_mask, loss_calculator, self.test_step)
        return loss

    def step_trough_batches(self, data_loader, mask, step_function) -> Dict[str, float]:
        running_loss = {}
        batch_index = 1
        for batch_index, batch in enumerate(data_loader):
            images, _ = batch
            inputs, targets = self.split_images_into_input_target(images)
            loss = step_function(inputs, targets, mask)
            for k, v in loss.items():
                running_loss[k] = running_loss.get(k, 0) + v
        return {k: v / batch_index for k, v in running_loss.items()}

    def train_step(self, inputs, targets, train_mask) -> Dict[str, float]:
        self.generator.train()
        self.generator_optimizer.zero_grad()
        generated = self.generator(inputs, targets, train_mask)[0]
        loss = self.generator_loss_function(generated, targets)
        loss = loss.sum(-1)
        loss = (loss * train_mask).sum(0)
        loss = loss.mean()
        loss.backward()
        self.generator_optimizer.step()
        return {'mse': loss.item()}

    def test_step(self, inputs, targets, test_mask) -> Dict[str, float]:
        self.generator.eval()
        generated = self.generator(inputs, targets, test_mask)[0]
        loss = self.generator_loss_function(generated, targets)
        loss = loss.sum(-1)
        loss = (loss * test_mask).sum(0)
        loss = loss.mean()
        return {'mse': loss.item()}




class SWN(Model):

    def __post_init__(self):
        self.generator, self.generator_optimizer, self.generator_loss_function = self.build_generator()

    def build_generator(self):
        generator = StochasticWavenet(input_dim=self.input_dim, layer_dim=self.layer_dim,
                                      output_dim=self.input_dim, num_layers=self.num_layers,
                                      num_stochastic_layers=self.num_layers-1)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.learning_rate, eps=1e-5)
        generator.eval()
        return generator, generator_optimizer, torch.nn.MSELoss()

    def save_model(self, epoch):
        state = {
                'epoch': epoch,
                'generator': self.generator.state_dict(),
                'gen_optimizer': self.generator_optimizer.state_dict(),
                }
        torch.save(state, Path('models', f'{self.model_name}.pth'))

    def load_model(self):
        state = torch.load(Path('models', f'{self.model_name}.pth'), map_location='cpu')
        self.generator.load_state_dict(state['generator'])
        self.generator_optimizer.load_state_dict(state['gen_optimizer'])

    def train(self, train_data_loader: DataLoader, test_data_loader: DataLoader, epochs: int,
              patience: int):

        receptive_field = self.get_receptive_field()
        print(f"The receptive field is {receptive_field}")

        sequence_length = self.infer_sequence_length(test_data_loader)
        train_mask = self.create_mask(sequence_length, train_data_loader.batch_size, receptive_field)
        test_mask = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)

        best_loss_so_far = 9999999
        kld_weight = 0.005

        loss_stats = {'train_mse': [], 'test_mse': [], 'train_kld': [], 'test_kld': []}
        for epoch in range(epochs):
            train_loss = self.step_trough_batches(train_data_loader, train_mask, self.train_step,
                                                  kld_weight)
            test_loss = self.step_trough_batches(test_data_loader, test_mask, self.test_step,
                                                 kld_weight)

            loss_stats['train_mse'].append(train_loss['mse'])
            loss_stats['test_mse'].append(test_loss['mse'])
            loss_stats['train_kld'].append(train_loss['kld'])
            loss_stats['test_kld'].append(test_loss['kld'])

            test_loss_this_epoch = test_loss['mse'] + kld_weight * test_loss['kld']
            print(f"epoch {epoch} | train mse : {train_loss['mse']} kld: {train_loss['kld']} | " +
                  f"test mse: {test_loss['mse']} kld {test_loss['kld']} weighted: {test_loss_this_epoch}")
            if self.early_stopping(best_loss_so_far, test_loss_this_epoch, patience) or epoch == epochs - 1:
                print(f" Saving model at epoch {epoch} with loss {best_loss_so_far}")
                self.save_model(epoch)
                self.save_loss_stats(loss_stats)
                return epoch
            elif test_loss_this_epoch < best_loss_so_far:
                best_loss_so_far = test_loss_this_epoch

            kld_weight = self.get_new_kld_weight(epoch, total_epoch=epochs,
                                                 init_kd=kld_weight)
            print(f"New KLD weight: {kld_weight}")
            self.adjust_lr(self.generator_optimizer, epoch, total_epoch=epochs,
                           init_lr=self.learning_rate)

        self.save_model(epochs)

    def step_trough_batches(self, data_loader, mask,  step_function,
                            kld_weight) -> Dict[str, float]:
        running_loss = {}
        batch_index = 1
        for batch_index, batch in enumerate(data_loader):
            images, _ = batch
            inputs, targets = self.split_images_into_input_target(images)
            loss = step_function(inputs, targets, mask, kld_weight)
            for k, v in loss.items():
                running_loss[k] = running_loss.get(k, 0) + v
        return {k: v / batch_index for k, v in running_loss.items()}

    def train_step(self, inputs, targets, train_mask, kld_weight) -> Dict[str, float]:
        self.generator.train()
        self.generator_optimizer.zero_grad()
        outputs = self.generator(inputs, targets, train_mask)
        generated, kld_loss = outputs
        loss = self.generator_loss_function(generated, targets)
        loss = loss.sum(-1)
        loss = (loss * train_mask).sum(0)
        loss = loss.mean()
        total_loss = loss + kld_loss * kld_weight
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.1, 'inf')
        self.generator_optimizer.step()
        return {'mse': loss.item(), 'kld': kld_loss.item()}

    def test_step(self, inputs, targets, test_mask, kld_weight) -> Dict[str, float]:
        self.generator.eval()
        outputs = self.generator(inputs, targets, test_mask)
        generated, kld_loss = outputs[0], outputs[1]
        loss = self.generator_loss_function(generated, targets)
        loss = loss.sum(-1)
        loss = (loss * test_mask).sum(0)
        loss = loss.mean() + kld_loss * kld_weight
        return {'mse': loss.item(), 'kld': kld_loss.item()}

    @staticmethod
    def adjust_lr(optimizer, epoch, total_epoch, init_lr):
        lr = init_lr * (0.5 * (1 + math.cos(math.pi * float(epoch) / total_epoch)))
        print(f"New learning rate {lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def get_new_kld_weight(epoch, total_epoch, init_kd, end_kd=1):
        return end_kd + (init_kd - end_kd) * math.cos(0.5 * math.pi * float(epoch) / total_epoch)


