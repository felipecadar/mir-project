import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import math
import torchaudio

class Discriminator(nn.Module):
    def __init__(self, l_in=44101, n_classes=5, latent=200):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(n_classes, latent)

        self.net = nn.Sequential(
                        nn.Linear(l_in + latent, l_in // 2),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 2, l_in // 4),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 4, l_in // 8),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 8, 100),
                        nn.LeakyReLU(),
                        nn.Linear(100, 1),
                        )

    def forward(self, x, context):
        x = torch.cat([x, self.emb (context)], -1)
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, l_in=44101, n_classes=5, latent=200):
        super(Generator, self).__init__()
        self.emb = nn.Embedding(n_classes, latent)

        self.net = nn.Sequential(
                        nn.Linear(l_in + latent, l_in // 2),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 2, l_in // 4),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 4, l_in // 8),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 8, l_in // 8),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 8, l_in // 4),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 4, l_in // 2),
                        nn.LeakyReLU(),
                        nn.Linear(l_in // 2, l_in),                        
                        )

    def forward(self, x, context): 
        x = torch.cat([x,  self.emb(context)], -1)
        return self.net(x)