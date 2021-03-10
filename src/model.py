import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import math
class EffectModel(nn.Module):
    def __init__(self, n, sr, filter_size=1):
        super(EffectModel, self).__init__()
        seq = []

        for i in range(n):
            seq.append(nn.Conv1d(1, 3, filter_size*sr, padding=int(((filter_size*sr) -1 ) / 2)))
            seq.append(nn.Conv1d(3, 3, filter_size*sr, padding=int(((filter_size*sr) -1 ) / 2)))
            seq.append(nn.Conv1d(3, 1, filter_size*sr, padding=int(((filter_size*sr) -1 ) / 2)))

        self.pedal = nn.Sequential(*seq)

        self._initialize_weights()

    def forward(self, x):
        return self.pedal(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()