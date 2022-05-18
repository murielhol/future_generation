from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid, tanh


class Wavenet(nn.Module):
    def __init__(self, input_dim, layer_dim, output_dim, num_layers: int):
        super(Wavenet, self).__init__()
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.num_layer = num_layers
        self.embedding = nn.Conv1d(input_dim, layer_dim, kernel_size=1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.skip.append(nn.Conv1d(layer_dim, layer_dim, kernel_size=1))

        dilate = 1
        self.kernel_size = 2
        for i in range(self.num_layer):
            self.filter_convs.append(nn.Conv1d(layer_dim, layer_dim, kernel_size=self.kernel_size, dilation=dilate))
            self.gate_convs.append(nn.Conv1d(layer_dim, layer_dim, kernel_size=self.kernel_size, dilation=dilate))
            self.residuals.append(nn.Conv1d(layer_dim, layer_dim, kernel_size=1))
            self.skip.append(nn.Conv1d(layer_dim, layer_dim, kernel_size=1))

            dilate *= 2

        self.final1 = nn.Conv1d(layer_dim, layer_dim, kernel_size=1)
        self.final2 = nn.Conv1d(layer_dim, layer_dim, kernel_size=1)
        self.outputs = nn.Linear(layer_dim, output_dim)

    def forward(self, x, apply_tanh=True) -> List[torch.Tensor]:
        # x is [N, seq_length, channels]
        # but conv1d wants [N, channels, seq_length]
        x = x.permute(0, 2, 1)
        # embedding
        x = F.relu(self.embedding(x))
        final = self.skip[0](x)
        dilate = 1

        # wavenet forward
        for i in range(self.num_layer):
            conv_x = x.unsqueeze(-1)
            conv_x = F.pad(conv_x, (0, 0, dilate, 0))
            conv_x = conv_x.squeeze(-1)
            tanh_x = self.filter_convs[i](conv_x)
            sigmoid_x = self.gate_convs[i](conv_x)
            residual_x = self.residuals[i](x)

            sigmoid_x = sigmoid(sigmoid_x)
            tanh_x = tanh(tanh_x)

            x = tanh_x * sigmoid_x
            skip_x = self.skip[i + 1](x)

            x = skip_x + residual_x
            final = skip_x + final
            dilate *= 2

        # wavenet output_dim to fcl
        final = self.final1(F.relu(final))
        final = self.final2(F.relu(final))
        # make sure last dimension is number of channels
        final = final.permute(0, 2, 1)
        outputs = self.outputs(final)
        if apply_tanh:
            outputs = tanh(outputs)
        return [outputs]







