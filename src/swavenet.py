from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import sigmoid, tanh


def gaussian_kld(left, right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    mu_left, logvar_left = left; mu_right, logvar_right = right
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (torch.exp(logvar_left) / torch.exp(logvar_right)) +
                        ((mu_left - mu_right)**2.0 / torch.exp(logvar_right)) - 1.0)
    return gauss_klds


class WaveNetGate(nn.Module):

    def __init__(self, input_dim, output_dim, dilate):
        super(WaveNetGate, self).__init__()
        self.filter_conv = nn.Conv1d(input_dim, output_dim, 2, dilation=dilate)
        self.gate_conv = nn.Conv1d(input_dim, output_dim, 2, dilation=dilate)
        self.residual = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, inputs):

        conv_x, x = inputs
        tanh_x = self.filter_conv(conv_x)
        sigmoid_x = self.gate_conv(conv_x)
        residual_x = self.residual(x)
        sigmoid_x = sigmoid(sigmoid_x)
        tanh_x = tanh(tanh_x)
        x = tanh_x * sigmoid_x
        return x + residual_x


class Gates(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_dim):
        super(Gates, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp_dim = mlp_dim
        self.fc1 = nn.Conv1d(input_dim, mlp_dim, 1)
        self.fc2 = nn.Conv1d(mlp_dim, output_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        return x


class StochasticWavenet(nn.Module):
    def __init__(self, input_dim, layer_dim, output_dim, num_layers, num_stochastic_layers):
        super(StochasticWavenet, self).__init__()
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.num_stochastic_layers = num_stochastic_layers
        self.num_layers = num_layers
        self.z_dim = layer_dim

        # first embed the input into layer dimensions, and add to residual with skip connection
        self.embedding = nn.Conv1d(input_dim, layer_dim, 1)
        self.initial_skip = Gates(layer_dim, layer_dim, layer_dim)

        dilate = 1

        # then forwards to non stochastic layers which is
        # a series of wavenet gates, and output goes to skip connection
        self.wavenet_gates_non_stochastic = nn.ModuleList()
        self.skip_non_stochastic = nn.ModuleList()

        for i in range(self.num_layers - self.num_stochastic_layers):
            self.wavenet_gates_non_stochastic.append(WaveNetGate(layer_dim, layer_dim, dilate))
            self.skip_non_stochastic.append(Gates(layer_dim, layer_dim, layer_dim))
            dilate *= 2

        # feed the targets (aka 'the future') to the backwards pass to
        # get a posterior distribution over z. This is a series of
        # wavenet gates and normal gates
        self.backward_wavenet_gates = nn.ModuleList()
        self.backward_gates = nn.ModuleList()

        # then forwards to stochastic layers which is
        # wavenet gate, then normal gates to get a prior over Z
        self.wavenet_gates_stochastic = nn.ModuleList()
        self.gates_stochastic = nn.ModuleList()
        self.skip_gates = nn.ModuleList()
        self.prior_gates = nn.ModuleList()
        # the inference gates combine forwards and backwards vectors into a posterior over z
        self.inference_gates = nn.ModuleList()

        for i in range(self.num_stochastic_layers):

            self.backward_wavenet_gates.append(WaveNetGate(layer_dim, layer_dim, dilate))
            self.backward_gates.append(Gates(layer_dim * 2, layer_dim, layer_dim))

            self.wavenet_gates_stochastic.append(WaveNetGate(layer_dim, layer_dim, dilate))
            self.prior_gates.append(Gates(layer_dim, layer_dim * 2, layer_dim))
            self.gates_stochastic.append(Gates(layer_dim + layer_dim, layer_dim, layer_dim))
            self.inference_gates.append(Gates(layer_dim * 2, layer_dim * 2, layer_dim))
            self.skip_gates.append(Gates(layer_dim + layer_dim, layer_dim, layer_dim))

            dilate *= 2

        self.final_dilate = dilate // 2
        self.final1 = nn.Conv1d(layer_dim, layer_dim, 1)
        self.final2 = nn.Conv1d(layer_dim, layer_dim, 1)
        self.outputs = nn.Linear(layer_dim, output_dim)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def backward_pass(self, x) -> List[torch.Tensor]:
        x = self.embedding(x)
        target = x
        backwards = []
        dilate = self.final_dilate
        for i in range(self.num_stochastic_layers):

            conv_x = x.unsqueeze(-1)
            conv_x = F.pad(conv_x, (0, 0, 0, dilate))
            conv_x = conv_x.squeeze(-1)
            inputs = [conv_x, x]
            conv_x = self.backward_wavenet_gates[self.num_stochastic_layers - 1 - i](inputs)
            x = conv_x
            backward_vec = torch.cat([conv_x, target], 1)

            backward_vec = self.backward_gates[i](backward_vec)
            backwards.append(backward_vec)

            if dilate > 1:
                dilate = dilate // 2

        return backwards

    def forward(self, x, y=None, mask=None) -> List[torch.Tensor]:
        # x, y are [N, seq_length, channels]
        # but conv1d wants [N, channels, seq_length]
        x = x.permute(0, 2, 1)     

        # when you are only interested in generating,
        # targets are not needed, and so posterior is not computed
        targets_provided = y is not None
        if targets_provided:
            y = y.permute(0, 2, 1)
            backward_vecs = self.backward_pass(y)
        else:
            backward_vecs = []

        x = F.relu(self.embedding(x))

        # forwards non stochastic layers
        dilate = 1
        final = self.initial_skip(x)
        for i in range(self.num_layers - self.num_stochastic_layers):
            conv_x = x.unsqueeze(-1)
            conv_x = F.pad(conv_x, (0, 0, dilate, 0))
            conv_x = conv_x.squeeze(-1)
            inputs = [conv_x, x]
            next_x = self.wavenet_gates_non_stochastic[i](inputs)
            final = final + self.skip_non_stochastic[i](next_x)
            x = next_x
            dilate *= 2

        # forwards stochastic layers
        kld_loss = 0
        for i in range(self.num_stochastic_layers):
            conv_x = x.unsqueeze(-1)
            conv_x = F.pad(conv_x, (0, 0, dilate, 0))
            conv_x = conv_x.squeeze(-1)
            inputs = [conv_x, x]
            next_x = self.wavenet_gates_stochastic[i](inputs)

            z_prior = self.prior_gates[i](next_x)
            z_prior = torch.clamp(z_prior, -8., 8.)
            mu_prior, theta_prior = torch.chunk(z_prior, 2, 1)

            # sample z from the prior
            # is replaced with z from posterior later on if training
            z = self.reparameterize(mu_prior, theta_prior)

            if backward_vecs:
                z_posterior = backward_vecs[self.num_stochastic_layers - 1 - i]
                z_posterior = torch.cat([next_x, z_posterior], 1)
                z_posterior = self.inference_gates[i](z_posterior)
                z_posterior = torch.clamp(z_posterior, max=8.)
                mu_posterior, theta_posterior = torch.chunk(z_posterior, 2, 1)

                '''
                trick from Fraccaro et al : https://arxiv.org/abs/1605.07571
                So that the model doesnt ignore the latent variables and just updates the
                backwards vectors to imitate the prior, 
                '''

                mu_posterior = mu_posterior + mu_prior

                # during the training step, you update the posterior
                if self.training:
                    z = self.reparameterize(mu_posterior, theta_posterior)

                kld_loss_this_layer = gaussian_kld([mu_posterior, theta_posterior], [mu_prior, theta_prior])
                # mask is [N, sequence length, dims]
                kld_loss_this_layer = kld_loss_this_layer.permute(0, 2, 1)
                kld_loss_this_layer = (kld_loss_this_layer.sum(-1) * mask).sum(0)
                kld_loss += kld_loss_this_layer.mean()

            tmp = torch.cat([next_x, z], 1)
            tmp = self.skip_gates[i](tmp)
            final = final + tmp
            next_x = torch.cat([next_x, z], 1)
            x = self.gates_stochastic[i](next_x)
            if i >= 0:
                dilate *= 2

        final = self.final1(F.relu(final))
        final = self.final2(F.relu(final))
        # make sure last dimension is number of channels
        final = final.permute(0, 2, 1)
        outputs = tanh(self.outputs(final))
        return [outputs, kld_loss]
