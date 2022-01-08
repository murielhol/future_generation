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
    WASSERSTEIN = "Wasserstein"


class ModelTypes:
    WN = 'wn'
    SWV = 'swn'
    DWN = 'dwn'
    SDWN = 'sdwn'


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
