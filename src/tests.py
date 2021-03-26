import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import librosa as lr
from siren import Sine


A = torch.rand(16, 1, 1025, 87)

mi = A.min(1, keepdim=True)[0]
ma = A.max(1, keepdim=True)[0]

print(mi.shape)
print(ma.shape)