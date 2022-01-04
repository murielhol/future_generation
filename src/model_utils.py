import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dataclasses import dataclass


class ModelRole:
    GENERATOR = 'generator'
    DISCRIMINATOR = 'discriminator'


class LossFunctions:
    MSE = 'MSE'
    GAUSSIAN = 'Gaussian'
    SOFTMAX = 'Softmax'
    WASSERSTEIN = "Wasserstein"


class ModelTypes:
    WN = 'wn'
    SWV = 'swn'
    DWN = 'dwn'
    SDWN = 'sdwn'


class Regressor(nn.Module):

    def __init__(self, loss_function, layer_dim, output_dim):
        super(Regressor, self).__init__()
        self.loss_function = loss_function
        self.output_dim = output_dim

        if self.loss_function == LossFunctions.MSE:
            self.mean = nn.Linear(layer_dim, self.output_dim)

        if self.loss_function == LossFunctions.GAUSSIAN:
            self.mean = nn.Linear(layer_dim, self.output_dim)
            self.var = nn.Linear(layer_dim, self.output_dim)

        if self.loss_function == LossFunctions.SOFTMAX:
            self.K = int(self.loss_function.split('@')[-1])
            self.softmax = nn.Linear(layer_dim, self.output_dim * self.K)

        if self.loss_function == LossFunctions.WASSERSTEIN:
            self.mean = nn.Linear(layer_dim, self.output_dim)

    def forward(self, inputs):

        if self.loss_function == LossFunctions.MSE:
            mean = self.mean(inputs)
            return [mean]

        if self.loss_function == LossFunctions.GAUSSIAN:
            return [self.mean(inputs), self.var(inputs)]

        if self.loss_function == LossFunctions.SOFTMAX:
            seqlen, batch_size, _ = inputs.size()
            predict = self.softmax(inputs).view(seqlen * batch_size * self.output_dim, self.K)
            predict = F.log_softmax(predict)
            predict = predict.view(seqlen, batch_size, self.output_dim, self.K)
            return [predict]

        if self.loss_function == LossFunctions.WASSERSTEIN:
            return self.mean(inputs)


@dataclass
class LossCalculator:

    loss_function: str

    def calculate_loss(self, target, outputs):

        '''
        make sure that whatever it returns, you want to
        minimize it
        '''

        if self.loss_function == LossFunctions.MSE:
            mean = outputs[0]
            loss = (mean - target).pow(2)
            return loss

        if self.loss_function == LossFunctions.GAUSSIAN:
            mean = outputs[0]
            logvar = outputs[1]
            logvar = torch.clamp(logvar, -12., 12.)
            var = torch.exp(logvar)
            diff = target - mean
            res = -torch.pow(diff, 2) / (2 * torch.pow(var, 2))
            res = 0.5 * math.log(2 * math.pi) - logvar + res
            return res
