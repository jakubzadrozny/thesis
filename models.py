import os.path

import torch
from torch import nn
import torch.nn.functional as F

from pointnet import PointNetfeat

LATENT = 128
ENCODER_HIDDEN = 1024
OUT_POINTS = 2048
DECODER_LAYERS = [LATENT, 512, 1024, 1024, 2048, 3*OUT_POINTS]

MODELS_DIR = 'trained'
MODELS_EXT = '.dms'
MODEL_DEFAULT_NAME = 'trained'

if torch.cuda.is_available():
    from chamfer_distance import ChamferDistance
    cdist = ChamferDistance()
    def cd(x, y):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        d1, d2 = cdist(x, y)
        return torch.sum(d1, dim=1) + torch.sum(d2, dim=1)
else:
    def cd(S, T):
        S = S.permute(0, 2, 1).unsqueeze(2)
        T = T.permute(0, 2, 1).unsqueeze(1)

        # S_center = S.mean(dim=1, keepdim=True)
        # T_center = T.mean(dim=1, keepdim=True)

        d = torch.sum(torch.pow(S - T, 2), dim=3)
        # d1_center = torch.sum(torch.pow(S - S_center, 2), dim=3)
        # d2_center = torch.sum(torch.pow(T - T_center, 2), dim=3)

        # d1 = torch.sum( d1_center * torch.min(d, dim=2)[0], dim=1 )
        # d2 = torch.sum( d2_center * torch.min(d, dim=1)[0], dim=1 )
        d1 = torch.sum(torch.min(d, dim=2)[0], dim=1)
        d2 = torch.sum(torch.min(d, dim=1)[0], dim=1)

        return d1+d2


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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z_mean, z_log_sigma2, shape):
        z = self.sample(z_mean, z_log_sigma2)
        rec = self.decoder(z).view(shape)
        return rec

    def forward(self, x):
        z_mean, z_log_sigma2 = self.encode(x)
        rec = self.decode(z_mean, z_log_sigma2, x.shape)
        return rec, z_mean, z_log_sigma2

    def elbo_loss(self, x, mc_samples=1, beta=1.0, lbd=0.0):
        N = x.size(0)
        z_mean, z_log_sigma2 = self.encode(x)
        KL_loss = torch.max(
            torch.tensor(lbd),
            (-0.5 / N) * torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp())
        )

        rec_loss = 0
        for i in range(mc_samples):
            rec = self.decode(z_mean, z_log_sigma2, x.shape)
            rec_loss += torch.mean(cd(x, rec))
        rec_loss /= mc_samples

        return (rec_loss + beta * KL_loss,
                {'rec': rec_loss.item(), 'KL': KL_loss.item()})

    def save_to_drive(self, name=MODEL_DEFAULT_NAME):
        torch.save(self.state_dict(), os.path.join(MODELS_DIR, name+MODELS_EXT))

    @staticmethod
    def load_from_drive(hidden, decoder_dims, name=MODEL_DEFAULT_NAME):
        model = VAE(hidden, decoder_dims)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT)))
        model.eval()
        return model
