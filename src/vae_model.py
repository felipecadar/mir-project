import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import librosa as lr

# Based on https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 256 # latent dimension for sampling

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=0
        )

        self.enc5 = nn.Conv2d(
            in_channels=init_channels*8, out_channels=init_channels*16, kernel_size=kernel_size, 
            stride=2, padding=0
        )

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(3840, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 3840)

        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=init_channels*16, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )

        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        # print("enc1", x.shape)
        x = F.relu(self.enc2(x))
        # print("enc2", x.shape)
        x = F.relu(self.enc3(x))
        # print("enc3", x.shape)
        x = F.relu(self.enc4(x))
        # print("enc4", x.shape)
        x = F.relu(self.enc5(x))
        # print("enc5", x.shape)


        batch, _, _, _ = x.shape
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

        z = z.view(-1, 128, 30, 1)
        # print("view", z.shape)
    
        # decoding
        x = F.relu(self.dec1(z))
        # print("dec1", x.shape)
        x = F.relu(self.dec2(x))
        # print("dec2", x.shape)
        x = F.relu(self.dec3(x))
        # print("dec3", x.shape)
        x = F.relu(self.dec4(x))
        # print("dec4", x.shape)
        x = F.relu(self.dec5(x))
        # print("dec5", x.shape)

        x = F.interpolate(x, size=[1025, 87])
        # print("Interpolate", x.shape)
        
        reconstruction = torch.sigmoid(x)
        # reconstruction = x
        return reconstruction, mu, log_var


if __name__ == "__main__":
    model = ConvVAE()

    x = torch.rand(1, 1, 1025, 87)
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
