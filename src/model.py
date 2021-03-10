import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models

class EffectModel(nn.Module):
    def __init__(self, n, sr, filter_size=1):
        super(EffectModel, self).__init__()
        self.pedal = nn.Sequential(*[nn.Conv1d(1, 1, filter_size*sr, padding=int(((filter_size*sr) -1 ) / 2)) for i in range(n)])


    def forward(self, x):
        return self.pedal(x)