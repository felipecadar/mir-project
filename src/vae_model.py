import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import librosa as lr
from siren import Sine

# Based on https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

kernel_size = 5 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 1024 # latent dimension for sampling

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        self.sine = Sine(w0=1)

        # encoder
        self.enc1 = nn.Conv1d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=3, padding=1
        )
        self.enc2 = nn.Conv1d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=3, padding=1
        )
        self.enc3 = nn.Conv1d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=3, padding=1
        )
        self.enc4 = nn.Conv1d(
            in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=3, padding=0
        )

        self.enc5 = nn.Conv1d(
            in_channels=init_channels*8, out_channels=init_channels*16, kernel_size=kernel_size, 
            stride=3, padding=0
        )

        self.enc6 = nn.Conv1d(
            in_channels=init_channels*16, out_channels=init_channels*32, kernel_size=kernel_size, 
            stride=3, padding=0
        )

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(15104, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 15104)

        # decoder 
        self.dec1 = nn.ConvTranspose1d(
            in_channels=init_channels*32, out_channels=init_channels*16, kernel_size=kernel_size, 
            stride=3, padding=0
        )
        self.dec2 = nn.ConvTranspose1d(
            in_channels=init_channels*16, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=3, padding=0
        )
        self.dec3 = nn.ConvTranspose1d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=3, padding=0
        )
        self.dec4 = nn.ConvTranspose1d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=3, padding=1
        )
        self.dec5 = nn.ConvTranspose1d(
            in_channels=init_channels*2, out_channels=init_channels, kernel_size=kernel_size, 
            stride=3, padding=1
        )

        self.dec6 = nn.ConvTranspose1d(
            in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size, 
            stride=3, padding=1
        )

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = self.sine(self.enc1(x))
        # print("enc1", x.shape)
        x = self.sine(self.enc2(x))
        # print("enc2", x.shape)
        x = self.sine(self.enc3(x))
        # print("enc3", x.shape)
        x = self.sine(self.enc4(x))
        # print("enc4", x.shape)
        x = self.sine(self.enc5(x))
        # print("enc5", x.shape)
        x = self.sine(self.enc6(x))
        # print("enc6", x.shape)


        batch, _, _, = x.shape
        # x = F.adaptive_avg_pool2d(x, 1).view(batch, -1)
        x = x.view(batch, -1)

        # print("view", x.shape)

        hidden = self.fc1(x)
        # print("hidden", x.shape)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # print("mu", mu.shape)
        # print("logvar", log_var.shape)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # print("reparameterize", z.shape)
        z = self.fc2(z)
        # print("fc2", z.shape)

        z = z.view(batch, init_channels*32, -1)
        # print("view", z.shape)
    
        # decoding
        x = self.sine(self.dec1(z))
        # print("dec1", x.shape)
        x = self.sine(self.dec2(x))
        # print("dec2", x.shape)
        x = self.sine(self.dec3(x))
        # print("dec3", x.shape)
        x = self.sine(self.dec4(x))
        # print("dec4", x.shape)
        x = self.sine(self.dec5(x))
        # print("dec5", x.shape)
        x = self.sine(self.dec6(x))
        # print("dec6", x.shape)

        x = F.interpolate(x, size=[44101])
        # print("Interpolate", x.shape)
        
        reconstruction = self.sine(x)
        # reconstruction = x
        return reconstruction, mu, log_var


if __name__ == "__main__":
    model = ConvVAE()

    # x = torch.rand(1, 1, 1025, 87)

    x = torch.rand(1, 1, 44101)
    print("INIT SHAPE", x.shape)

    reconstruction, mu, log_var = model(x)
    print("FINAL SHAPE", reconstruction.shape)

