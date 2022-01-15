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
    SWN = 'swn'
    DWN = 'dwn'
    SDWN = 'sdwn'
