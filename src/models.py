import math
from typing import Dict
import torch
from torch import autograd

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
        with open(Path('outputs', f'{self.model_name}', 'loss_stats.json'), 'w') as fp:
            json.dump(loss_stats, fp)

    def get_receptive_field(self) -> int:
        sequence_length = 1000
        # make sure that batch norm is turned off
        self.generator.eval()
        self.generator_optimizer.zero_grad()
        x = np.ones([1, sequence_length, self.input_dim])
        x = Variable(torch.from_numpy(x).float(), requires_grad=True)
        mask = self.create_mask(sequence_length, 1, sequence_length-1)
        generated = self.generator(x)[0]
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
        plt.savefig(Path('outputs', f'{self.model_name}', 'freerunning.png'))

    def infer_sequence_length(self, test_data_loader):
        test_iterator = iter(test_data_loader)
        sample_x, _ = next(test_iterator)
        assert len(sample_x.shape) == 4  # [batch_size, 1, seq_length, num_features]
        assert self.input_dim == sample_x.shape[3]
        sequence_length = sample_x.shape[2] - 1  # the input length is the number of rows - 1
        return sequence_length

    def free_running(self, images, receptive_field):
        sequence_length = images.shape[1]
        for i in range(sequence_length - receptive_field):
            sample_input = images[:, i:i+receptive_field, :]
            sample_output = self.generator(sample_input)[0]
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
        mask[:, receptive_field:] = 1
        return mask



class WN(Model):

    def __post_init__(self):
        self.generator, self.generator_optimizer, self.generator_loss_function = self.build_generator()

    def build_generator(self):
        generator = Wavenet(input_dim=self.input_dim, layer_dim=self.layer_dim, output_dim=self.input_dim,
                            num_layers=self.num_layers)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.learning_rate, eps=1e-5)
        generator.eval()
        return generator, generator_optimizer, torch.nn.MSELoss(reduction='none')

    def save_model(self, epoch, final=False):
        state = {
                'epoch': epoch,
                'generator': self.generator.state_dict(),
                'gen_optimizer': self.generator_optimizer.state_dict(),
                }
        name = f'state.pth' if final else f'state_{epoch}.pth'
        torch.save(state, Path('outputs', self.model_name, name))

    def load_model(self):
        state = torch.load(Path('outputs', self.model_name, f'state.pth'), map_location='cpu')
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
        loss_stats = {'train_mse': [], 'test_mse': []}

        epoch = 0
        self.save_model(epoch)
        for epoch in range(epochs):

            train_loss = self.step_trough_batches(train_data_loader, train_mask, self.train_step)
            test_loss = self.step_trough_batches(test_data_loader, test_mask, self.test_step)

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
        loss = self.step_trough_batches(test_data_loader, test_mask, self.test_step)
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
        generated = self.generator(inputs)[0]
        loss = self.generator_loss_function(generated, targets)
        loss = loss.sum(-1)
        loss = (loss * train_mask).mean()
        loss.backward()
        self.generator_optimizer.step()
        return {'mse': loss.item()}

    def test_step(self, inputs, targets, test_mask) -> Dict[str, float]:
        self.generator.eval()
        generated = self.generator(inputs)[0]
        loss = self.generator_loss_function(generated, targets)
        loss = loss.sum(-1)
        loss = (loss * test_mask).mean()
        return {'mse': loss.item()}


class SWN(Model):

    def __post_init__(self):
        self.generator, self.generator_optimizer, self.generator_loss_function = self.build_generator()

    def build_generator(self):
        generator = StochasticWavenet(input_dim=self.input_dim, layer_dim=self.layer_dim,
                                      output_dim=self.input_dim, num_layers=self.num_layers-1,
                                      num_stochastic_layers=1)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.learning_rate, eps=1e-5)
        generator.eval()
        return generator, generator_optimizer, torch.nn.MSELoss(reduction='none')

    def save_model(self, epoch, final=False):
        state = {
                'epoch': epoch,
                'generator': self.generator.state_dict(),
                'gen_optimizer': self.generator_optimizer.state_dict(),
                }
        name = f'state.pth' if final else f'state_{epoch}.pth'
        torch.save(state, Path('outputs', self.model_name, name))

    def load_model(self):
        state = torch.load(Path('outputs', self.model_name, f'state.pth'), map_location='cpu')
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
                if epoch % 10 == 0: # save the model every 10 epochs
                    self.save_model(epoch, final=False)
                    self.save_loss_stats(loss_stats)

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
        loss = (loss * train_mask).mean()
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
        loss = (loss * test_mask).mean()
        loss = loss + kld_loss * kld_weight
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

    def evaluate(self, test_data_loader: DataLoader):
        self.load_model()
        receptive_field = self.get_receptive_field()
        print(f"The receptive field is {receptive_field}")
        sequence_length = self.infer_sequence_length(test_data_loader)
        test_mask = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)
        loss = self.step_trough_batches(test_data_loader, test_mask, self.test_step, kld_weight=0)
        return loss

#
# class DSWN(Model):
#
#     '''
#     Stochastic Wavenet trained with a discriminator
#     '''
#
#     critic_updates = 5
#     discriminator: torch.nn.Module = field(init=False)
#     discriminator_optimizer: torch.optim = field(init=False)
#
#     def __post_init__(self):
#         self.generator, self.generator_optimizer, self.generator_loss_function = self.build_generator()
#         self.discriminator, self.discriminator_optimizer, self.discriminator_loss_function = self.build_generator()
#
#     def build_generator(self):
#         generator = StochasticWavenet(input_dim=self.input_dim, layer_dim=self.layer_dim,
#                                       output_dim=self.input_dim, num_layers=self.num_layers,
#                                       num_stochastic_layers=self.num_layers-1)
#         ''''
#         The learning rate of the generator has to be smaller than then the learning rate of the discriminator
#         to prevent that the generator is too good too early, as that would stop the learning process
#         '''
#         generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.learning_rate/10.0, eps=1e-5)
#         generator.eval()
#         return generator, generator_optimizer, torch.nn.MSELoss(reduction='none')
#
#     def build_discriminator(self):
#         discriminator = Wavenet(input_dim=self.input_dim, layer_dim=self.layer_dim,
#                                 output_dim=self.input_dim, num_layers=self.num_layers)
#         discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate, eps=1e-5)
#         discriminator.eval()
#         return discriminator, discriminator_optimizer, torch.nn.MSELoss(reduction='none')
#
#     def save_model(self, epoch, final=False):
#         state = {
#                 'epoch': epoch,
#                 'generator': self.generator.state_dict(),
#                 'gen_optimizer': self.generator_optimizer.state_dict(),
#                 'discriminator': self.discriminator.state_dict(),
#                 'discriminator_optimizer': self.discriminator_optimizer.state_dict()
#                 }
#         name = f'state.pth' if final else f'state_{epoch}.pth'
#         torch.save(state, Path('outputs', self.model_name, name))
#
#     def load_model(self):
#         state = torch.load(Path('models', f'{self.model_name}.pth'), map_location='cpu')
#         self.generator.load_state_dict(state['generator'])
#         self.generator_optimizer.load_state_dict(state['gen_optimizer'])
#         self.discriminator.load_state_dict(state['discriminator'])
#         self.discriminator_optimizer.load_state_dict(state['discriminator_optimizer'])
#
#     def train(self, train_data_loader: DataLoader, test_data_loader: DataLoader, epochs: int,
#               patience: int):
#
#         receptive_field = self.get_receptive_field()
#         print(f"The receptive field is {receptive_field}")
#
#         sequence_length = self.infer_sequence_length(test_data_loader)
#         train_mask_generator = self.create_mask(sequence_length, train_data_loader.batch_size, receptive_field)
#         test_mask_generator = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)
#
#         train_mask_discriminator = self.create_mask(sequence_length, train_data_loader.batch_size, receptive_field)
#         test_mask_discriminator = self.create_mask(sequence_length, test_data_loader.batch_size, receptive_field)
#
#         best_loss_so_far = 9999999
#         kld_weight = 0.005
#
#         '''
#         When critic loss goes to 0, this indicates good quality samples
#         '''
#         loss_stats = {'train_mse': [], 'test_mse': [], 'train_kld': [], 'test_kld': []}
#         for epoch in range(epochs):
#             train_loss = self.step_trough_batches(train_data_loader, train_mask_generator, train_mask_discriminator,
#                                                   self.train_step, kld_weight)
#             test_loss = self.step_trough_batches(test_data_loader, test_mask_generator, test_mask_discriminator,
#                                                  self.test_step, kld_weight)
#
#             loss_stats['train_mse'].append(train_loss['mse'])
#             loss_stats['test_mse'].append(test_loss['mse'])
#             loss_stats['train_kld'].append(train_loss['kld'])
#             loss_stats['test_kld'].append(test_loss['kld'])
#
#             test_loss_this_epoch = test_loss['mse'] + kld_weight * test_loss['kld']
#             print(f"epoch {epoch} | train mse : {train_loss['mse']} kld: {train_loss['kld']} | " +
#                   f"test mse: {test_loss['mse']} kld {test_loss['kld']} weighted: {test_loss_this_epoch}")
#             if self.early_stopping(best_loss_so_far, test_loss_this_epoch, patience) or epoch == epochs - 1:
#                 print(f" Saving model at epoch {epoch} with loss {best_loss_so_far}")
#                 self.save_model(epoch)
#                 self.save_loss_stats(loss_stats)
#                 return epoch
#             elif test_loss_this_epoch < best_loss_so_far:
#                 best_loss_so_far = test_loss_this_epoch
#
#             kld_weight = self.get_new_kld_weight(epoch, total_epoch=epochs,
#                                                  init_kd=kld_weight)
#             print(f"New KLD weight: {kld_weight}")
#             self.adjust_lr(self.generator_optimizer, epoch, total_epoch=epochs,
#                            init_lr=self.learning_rate)
#
#         self.save_model(epochs)
#
#     def calc_gradient_penalty(self, real_data, fake_data):
#         '''
#         Arjovsky et al. (2017) provide a proof that the EM-distance is equal to the
#         Wasserstein distance if the critic obeys two constraints. Firstly it has to be
#         continuous and differentiable, as is the case for current architectures of neural networks.
#         Secondly, it has to be a member of 1-Lipschitz functions, which is explained in detail in
#         Arjovsky et al. (2017).
#         In practice, Arjovsky et al. (2017) enforce the Lipschitz constraint by clipping the
#         parameters of the critic between a specific range (Arjovsky et al., 2017). Gulrajani et al. (2017)
#         suggest to replace the parameter clipping with a gradient penalty that penalizes for a
#         gradient norm different than 1. They show that this softer constraint improves on the WGAN
#         training for tasks of language and image generation. This alters the critic loss to Eq. 2.15,
#         that shows the coupled two-sided gradient penalty. For a compact overview of all possible
#         adversaries and gradient penalties, we refer to Dong and Yang (2019).
#         '''
#         batch_size = real_data.size[0]
#         alpha = torch.rand(batch_size, 1)
#         alpha = alpha.expand(real_data.size())
#
#         interpolates = alpha * real_data + (1 - alpha) * fake_data
#
#         interpolates = autograd.Variable(interpolates, requires_grad=True)
#
#         disc_interpolates = self.discriminator(interpolates)
#
#         gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                                   grad_outputs= torch.ones(disc_interpolates.size()),
#                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#         gradients = gradients[-1,:,:]
#         gradients = gradients.view(batch_size, -1)
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 1
#
#         return gradient_penalty
#
#     def step_trough_batches(self, data_loader, mask_generator, mask_discriminator, step_function,
#                             kld_weight) -> Dict[str, float]:
#
#         running_loss = {}
#         batch_index = 1
#         receptive_field = self.get_receptive_field()
#
#         self.generator.eval()
#         self.discriminator.train()
#         for batch_index, batch in enumerate(data_loader):
#             images, _ = batch
#             inputs, targets = self.split_images_into_input_target(images)
#             # As opposed to the vanilla GAN, in the WGAN setup it is beneficial to have a strong critic.
#             # Therefore, the critic is often trained more than the generator.
#             # Arjovsky et al. (2017) show that a WGAN trained with 1 generator train step per 5
#             # critic train steps can learn to generate natural images with state of the art sample quality.
#             critic_updates = 5
#             if (batch_index + 1) % critic_updates != 0:
#
#                 generated = self.generator(inputs)[0]
#
#                 '''
#                 The generator can generate multiple outputs in parallel
#                 here we need to consolidate them with their inputs before
#                 feeding to the discriminator
#                 '''
#                 input_len = images.shape[1]
#                 generated_len = input_len - receptive_field
#                 for i in range(generated_len):
#                     fake_pile = torch.cat((inputs[:, i + 1:i + receptive_field, :],
#                                            generated[:, i + receptive_field-1:i + receptive_field, :]))
#                     if i == 0:
#                         fake_stack = fake_pile
#                     else:
#                         fake_stack = torch.cat((fake_stack, fake_pile), dim=1)
#
#                 for i in range(generated_len):
#                     true_pile = torch.cat((inputs[i + 1:i + receptive_field, :, :],
#                                           inputs[i + receptive_field-1:i + receptive_field, :, :]))
#                     if i == 0:
#                         true_stack = true_pile
#                     else:
#                         true_stack = torch.cat((true_stack, true_pile), dim=1)
#
#                 fake_score = self.discriminator(fake_stack, apply_tanh=False)
#                 true_score = self.discriminator(true_stack, apply_tanh=False)
#
#                 d_loss = (mask_discriminator * true_score).mean() - (mask_discriminator * fake_score).mean()
#                 gradient_penalty = self.calc_gradient_penalty(inputs, generated)
#
#                 self.discriminator.zero_grad()
#                 d_loss_p = d_loss * mask_discriminator + gradient_penalty
#                 d_loss_p.backward()
#                 self.discriminator_optimizer.step()
#             else:
#
#                 self.generator.train()
#                 self.discriminator.eval()
#                 loss = step_function(inputs, targets, mask_generator, kld_weight)
#
#             for k, v in loss.items():
#                 running_loss[k] = running_loss.get(k, 0) + v
#         return {k: v / batch_index for k, v in running_loss.items()}
#
#     def train_step(self, inputs, targets, train_mask, kld_weight) -> Dict[str, float]:
#         self.generator.train()
#         self.generator_optimizer.zero_grad()
#         outputs = self.generator(inputs, targets, train_mask)
#         generated, kld_loss = outputs
#         loss = self.generator_loss_function(generated, targets)
#         loss = loss.sum(-1)
#         loss = (loss * train_mask).mean()
#         total_loss = loss + kld_loss * kld_weight
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.1, 'inf')
#         self.generator_optimizer.step()
#         return {'mse': loss.item(), 'kld': kld_loss.item()}
#
#     def test_step(self, inputs, targets, test_mask, kld_weight) -> Dict[str, float]:
#         self.generator.eval()
#         outputs = self.generator(inputs, targets, test_mask)
#         generated, kld_loss = outputs[0], outputs[1]
#         loss = self.generator_loss_function(generated, targets)
#         loss = loss.sum(-1)
#         loss = (loss * test_mask).mean()
#         loss = loss + kld_loss * kld_weight
#         return {'mse': loss.item(), 'kld': kld_loss.item()}
#
#     @staticmethod
#     def adjust_lr(optimizer, epoch, total_epoch, init_lr):
#         lr = init_lr * (0.5 * (1 + math.cos(math.pi * float(epoch) / total_epoch)))
#         print(f"New learning rate {lr}")
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#
#     @staticmethod
#     def get_new_kld_weight(epoch, total_epoch, init_kd, end_kd=1):
#         return end_kd + (init_kd - end_kd) * math.cos(0.5 * math.pi * float(epoch) / total_epoch)
