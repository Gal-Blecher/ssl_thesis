import torch
import torch.nn as nn
from torchvision.models import resnet18

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 32 * 32 * 3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        x = x.view(x.size(0), 3, 32, 32)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        self.z = self.reparameterize(mu, logvar)
        self.x_hat = self.decode(self.z)

        # Reconstruction loss
        self.recon_loss = nn.MSELoss(reduction='mean')(self.x_hat, x)

        # KL divergence loss
        self.kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())