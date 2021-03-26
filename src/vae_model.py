import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import librosa as lr
from siren import Sine
from sinelayer import SineLayer
from siren_pytorch import SirenNet, SirenWrapper

# Based on https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

hidden = 1024
latent_dim = 512 # latent dimension for sampling
n_layers = 3

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        self.encoder = SirenNet(44101, hidden, latent_dim, n_layers)
        
        self.mu = SineLayer(latent_dim, latent_dim)
        self.log_var = SineLayer(latent_dim, latent_dim)

        self.decoder = SirenNet(latent_dim, hidden, 44101, n_layers)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.encoder(x)
        
        mu = self.mu(x)
        log_var = self.log_var(x)

        x = self.reparameterize(mu, log_var)

        x = self.decoder(x)

    #     Yencmean=self.encodermean(x)
    #     Yencstd=self.encoderstd(x)
    #     #Yvariational= torch.normal(Yencmean, Yencstd)
    #     Yvariational= Yencmean + Yencstd*torch.randn_like(Yencstd)
    #     #for the randn_like see also: https://github.com/pytorch/examples/blob/master/vae/main.py
    #     Ypred=self.decoder(Yvariational)

    #     Ypred=F.interpolate(Ypred, size=[44101])

        return x, mu, log_var  

if __name__ == "__main__":
    model = ConvVAE()

    # x = torch.rand(1, 1, 1025, 87)

    x = torch.rand(1, 1, 44101)
    print("INIT SHAPE", x.shape)

    reconstruction, mu, log_var = model(x)
    print("FINAL SHAPE", reconstruction.shape)

