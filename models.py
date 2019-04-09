import os.path

import torch
from torch import nn
import torch.nn.functional as F

from pointnet import PointNetfeat


MODELS_DIR = 'trained'
MODELS_EXT = '.dms'
MODEL_DEFAULT_NAME = 'trained'

def cd(S, T):
    S = S.permute(0, 2, 1).unsqueeze(2)
    T = T.permute(0, 2, 1).unsqueeze(1)
    d = torch.sum(torch.pow(S - T, 2), dim=3)
    d1 = torch.sum(torch.min(d, dim=1)[0], dim=1)
    d2 = torch.sum(torch.min(d, dim=2)[0], dim=1)
    return d1+d2

def elbo_loss(x, reconstruction, z_mean, z_log_sigma2, beta=1.0):
    N = x.size(0)
    rec_loss = torch.mean(cd(x, reconstruction))
    KL_loss = (-0.5 / N) * (
        torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp()))

    return (rec_loss + beta * KL_loss,
            {'rec': rec_loss.item(), 'KL': KL_loss.item()})

class PointnetEncoder(nn.Module):
    def __init__(self, hidden, latent):
        super(PointnetEncoder, self).__init__()
        self.model = PointNetfeat()
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden, 2*latent)

    def forward(self, x):
        x = self.model(x)[0]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.chunk(x, 2, dim=-1)


class MLPDecoder(nn.Module):
    def __init__(self, dims):
        """dims[0] sould be latent dim, dims[-1] prod of origin dims"""
        super(MLPDecoder, self).__init__()
        self.fc = nn.ModuleList([])

        prev = dims[0]
        for nxt in dims[1:]:
            self.fc.append(nn.Linear(prev, nxt))
            prev = nxt

    def forward(self, x):
        for layer in self.fc[:-1]:
            x = F.relu(layer(x))
        return self.fc[-1](x)

class VAE(torch.nn.Module):
    def __init__(self, hidden, decoder_dims):
        super(VAE, self).__init__()
        self.encoder = PointnetEncoder(hidden, decoder_dims[0])
        self.decoder = MLPDecoder(decoder_dims)

    def sample(self, z_mean, z_log_sigma2):
        z_sigma = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_mean)
        epsilon.mul_(z_sigma)
        epsilon.add_(z_mean)
        return epsilon

    def forward(self, x):
        z_mean, z_log_sigma2 = self.encoder(x)
        z = self.sample(z_mean, z_log_sigma2)
        reconstruction = self.decoder(z).view(x.size())
        return reconstruction, z_mean, z_log_sigma2

    def save_to_drive(self, name=MODEL_DEFAULT_NAME):
        torch.save(self, os.path.join(MODELS_DIR, name+MODELS_EXT))

    @staticmethod
    def load_from_drive(name=MODEL_DEFAULT_NAME):
        return torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT))
