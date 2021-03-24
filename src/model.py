import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import math
import torchaudio
from unet_parts import *

def Conv1dSame(in_channels, out_channels, l_in, kernel_size, stride):
    pad = (stride*(l_in - 1) - l_in + kernel_size) / 2
    pad = math.ceil(pad)
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad)

class Discriminator(nn.Module):
    def __init__(self, l_in=44101, n_classes=5):
        super(Discriminator, self).__init__()

        self.emb = nn.Embedding(n_classes, l_in)

        self.net = nn.Sequential(
            # C64
            Conv1dSame(in_channels=2, out_channels=64,
                      l_in=l_in, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),

            # C128
            Conv1dSame(in_channels=64, out_channels=128,
                      l_in=l_in, kernel_size=5, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            # C256
            Conv1dSame(in_channels=128, out_channels=256,
                      l_in=l_in, kernel_size=5, stride=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            # C512
            Conv1dSame(in_channels=256, out_channels=512,
                      l_in=l_in, kernel_size=5, stride=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            # second last output layer
            Conv1dSame(in_channels=512, out_channels=512, kernel_size=5, stride=1, l_in=l_in),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            # patch output
            Conv1dSame(in_channels=512, out_channels=1, kernel_size=5, stride=1, l_in=l_in),
            # nn.Sigmoid(),

            nn.Linear(l_in, 512),
            nn.Linear(512, 1),
            # nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        cat = torch.cat([x1, self.emb(x2)], 1)
        y = self.net(cat)
        return y

class Generator(nn.Module):
    def __init__(self, n_channels=2, n_classes=5, l_in = 44101, bilinear=False):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.emb = nn.Embedding(n_classes, l_in)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

    def forward(self, x, context):
        x = torch.cat([x, self.emb(context)], 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    l_in = 44101
    x = torch.randn(1, 1, l_in)
    y = torch.randn(1, 1, l_in)
    ctx = torch.LongTensor([[3]])

    print("x", x.shape)
    print("y", y.shape)

    discriminator = Discriminator()
    d = discriminator(x, ctx)
    print("d",  d.shape)
    
    generator = Generator()
    g = generator(x, ctx)
    print("g",  g.shape)
    print("e",  g.shape)
