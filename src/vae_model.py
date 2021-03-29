import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import librosa as lr
from siren import Sine

# Based on https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

hidden = 512 # latent dimension for sampling
latent_dim = 256 # latent dimension for sampling

def final_loss(BCE, KLD, alpha=0.999):
    return (alpha * BCE) + (1-alpha) * KLD

def KLDloss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD
class Reshape(nn.Module):
  def __init__(self, a,b,c,d):
    super(Reshape, self).__init__()
    self.shape = (a,b,c,d)
  
  def forward(self, x):
    return x.view(*self.shape)

class Autoencoder(nn.Module):
    def __init__(self, H, W):
        super(Autoencoder, self).__init__()
        _W = (W//16) * 16
        _H = (H//16) * 16

        self.encoder = nn.Sequential(
            nn.Upsample(size=(_H,_W)),
            nn.Conv2d(1, 12, 3, stride=2, padding=1, bias = False), 
            nn.BatchNorm2d(12, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(24, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias = False), 
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias = False), 
            nn.BatchNorm2d(96, affine=False),
            nn.LeakyReLU(),            
            nn.Flatten(),
            nn.Linear((H//16)*(W//16)*96, hidden),
            nn.LeakyReLU()
        )

        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_log_var = nn.Linear(hidden, latent_dim)
        self.fc = nn.Linear(latent_dim, hidden)

        self.decoder = nn.Sequential(
            nn.Linear(hidden, (H//16)*(W//16)*96),
            nn.LeakyReLU(),
            Reshape(-1,96,H//16,W//16),
            nn.BatchNorm2d(96, affine=False),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(96, 48, 3, padding=1, bias = False), 
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU(),    
            nn.Upsample(scale_factor=2),
            nn.Conv2d(48, 24, 3, padding=1, bias = False), 
            nn.BatchNorm2d(24, affine=False),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(24, 12, 3, padding=1, bias = False),
            nn.BatchNorm2d(12, affine=False), 
            nn.LeakyReLU(),   
            nn.Upsample(size=(H,W)),
            nn.Conv2d(12, 1, 3, padding=1, bias = False), 
            nn.Tanh(),
            
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def latent(self, x):
        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        x = self.reparameterize(mu, log_var)
        return x

    def decode(self, x):
        x = self.fc(x)
        x = self.decoder(x)
        return x

    def forward(self, x, latent=False):
        x = self.encoder(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        x = self.reparameterize(mu, log_var)
        if latent:
            x_latent = x.detach().clone()

        x = self.fc(x)
        x = self.decoder(x)

        if latent:
            return x, x_latent
        else:
            return x, mu, log_var

        # return x, None,None

# define a Conv VAE

if __name__ == "__main__":
    # model = ConvVAE()
    model = Autoencoder(128, 87)

    x = torch.rand(1, 1, 128, 87)
    print("INIT SHAPE", x.shape)

    reconstruction, mu, log_var = model(x)
    print("FINAL SHAPE", reconstruction.shape)


# INIT SHAPE torch.Size([1, 1, 1025, 87])
# enc1 torch.Size([1, 8, 512, 43])
# enc2 torch.Size([1, 16, 256, 21])
# enc3 torch.Size([1, 32, 128, 10])
# enc4 torch.Size([1, 64, 63, 4])
# enc5 torch.Size([1, 128, 30, 1])
# view torch.Size([1, 3840])
# hidden torch.Size([1, 3840])
# mu torch.Size([1, 256])
# logvar torch.Size([1, 256])
# reparameterize torch.Size([1, 256])
# fc2 torch.Size([1, 3840])
# view torch.Size([1, 128, 30, 1])
# dec1 torch.Size([1, 64, 62, 4])
# dec2 torch.Size([1, 32, 126, 10])
# dec3 torch.Size([1, 16, 252, 20])
# dec4 torch.Size([1, 8, 504, 40])
# dec5 torch.Size([1, 1, 1008, 80])
# Interpolate torch.Size([1, 1, 1025, 87])
# FINAL SHAPE torch.Size([1, 1, 1025, 87])
