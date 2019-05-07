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
        # d1_cener = torch.sum(torch.pow(S - S_center, 2), dim=3)
        # d2_center = torch.sum(torch.pow(T - T_center, 2), dim=3)

        # d1 = torch.sum( d1_center * torch.min(d, dim=2)[0], dim=1 )
        # d2 = torch.sum( d2_center * torch.min(d, dim=1)[0], dim=1 )
        d1 = torch.sum(torch.min(d, dim=2)[0], dim=1)
        d2 = torch.sum(torch.min(d, dim=1)[0], dim=1)

        return d1+d2


class SimplePointnetEncoder(nn.Module):
    def __init__(self, hidden, latent):
        super().__init__()
        self.model = PointNetfeat()
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden, latent)

    def forward(self, x):
        x = F.relu(self.model(x)[0])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PointnetSoftmaxEncoder(SimplePointnetEncoder):
    def forward(self, x):
        x = super().forward(x)
        return F.softmax(x)


class ClassPointentEncoder(nn.Module):
    def __init__(self, hidden, K, latent):
        super().__init__()
        self.feats = PointNetfeat()
        self.fc1 = nn.Linear(1024 + K, hidden)
        self.fc2 = nn.Linear(hidden, latent)

    def forward(self, x, y):
        x = F.relu(self.model(x)[0])
        x = torch.concatenate((x, y), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, dims):
        """dims[0] sould be latent dim, dims[-1] prod of origin dims"""
        super().__init__()
        self.fc = nn.ModuleList([])

        prev = dims[0]
        for nxt in dims[1:]:
            self.fc.append(nn.Linear(prev, nxt))
            prev = nxt

    def forward(self, x):
        for layer in self.fc[:-1]:
            x = F.relu(layer(x))
        return self.fc[-1](x)


class VAE(nn.Module):
    def __init__(self, hidden, decoder_dims):
        super().__init__()
        latent = decoder_dims[0]
        self.mean_encoder = SimplePointnetEncoder(hidden, latent)
        self.sigma_encoder = SimplePointnetEncoder(hidden, latent)
        self.decoder = MLPDecoder(decoder_dims)

    def sample(self, z_mean, z_log_sigma2):
        z_sigma = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_mean)
        epsilon.mul_(z_sigma)
        epsilon.add_(z_mean)
        return epsilon

    def encode(self, x):
        return self.mean_encoder(x), self.sigma_encoder(x)

    def decode(self, z_mean, z_log_sigma2, shape):
        z = self.sample(z_mean, z_log_sigma2)
        rec = self.decoder(z)
        return rec.view(shape)

    def forward(self, x):
        z_mean, z_log_sigma2 = self.encode(x)
        rec = self.decode(z_mean, z_log_sigma2, x.shape)
        return rec, z_mean, z_log_sigma2

    def elbo_loss(self, x, mc_samples=1, lbd=0.0):
        z_mean, z_log_sigma2 = self.encode(x)
        KL_loss = torch.mean(
            torch.clamp(-0.5*(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp()),
                        min=lbd)
        )

        rec_loss = 0
        for i in range(mc_samples):
            rec = self.decode(z_mean, z_log_sigma2, x.shape)
            rec_loss += torch.mean(cd(x, rec))
        rec_loss /= mc_samples

        return (rec_loss + KL_loss,
                {'rec': rec_loss.item(), 'KL': KL_loss.item()})

    def save_to_drive(self, name=MODEL_DEFAULT_NAME):
        torch.save(self.state_dict(), os.path.join(MODELS_DIR, name+MODELS_EXT))

    @staticmethod
    def load_from_drive(hidden, decoder_dims, name=MODEL_DEFAULT_NAME):
        model = VAE(hidden, decoder_dims)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT)))
        model.eval()
        return model


class M2(nn.Module):
    def __init__(self, K, hidden, decoder_dims):
        super().__init__()
        latent = decoder_dims[0]
        decoder_dims[0] = latent + K
        self.num_classes = K
        self.class_encoder = SoftmaxPointnetEncoder(hidden, K)
        self.sigma_encoder = SimplePointnetEncoder(hidden, latent)
        self.mean_encoder = ClassPointnetEncoder(hidden, K, latent)
        self.decoder = MLPDecoder(decoder_dims)

    def encode_y(self, x):
        return self.class_encoder(x)

    def encode_z_based_on_y(self, x, y):
        z_log_sigma2 = self.sigma_encoder(x)
        z_mean = self.mean_encoder(x, y)
        return z_mean, z_log_sigma2

    def decode(self, z_mean, z_log_sigma2, y, shape):
        z = self.sample(z_mean, z_log_sigma2)
        latent = torch.cat((y, z), dim=1)
        rec = self.decoder(latent)
        return rec.view(shape)

    def elbo_known_y(self, x, y, alpha=0.1, lbd=0.0, mc_samples=1):
        z_mean, z_log_sigma2 = self.encode_based_on_y(x, y)
        KL_loss = torch.mean(
            torch.clamp(-0.5*(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp()),
                        min=lbd)
        )
        y_penalty = -torch.full_like(KL_loss, torch.log(1/K))

        rec_loss = 0
        for i in range(mc_samples):
            rec = self.decode(z_mean, z_log_sigma2, y, x.shape)
            rec_loss += torch.mean(cd(x, rec))
        rec_loss /= mc_samples

        if alpha > 0:
            y_pred = self.encode_y(x)
            clf_loss = -alpha*torch.mean(torch.log(torch.sum(y_pred * y, dim=1)))
        else:
            cls_loss = 0

        return rec_loss + KL_loss + y_penalty + clf_loss
                # {'rec': rec_loss.item(), 'KL': KL_loss.item(), 'clf_loss': clf_loss.item()})

    def elbo_unknown_y(self, x, lbd=0.0, mc_samples=1):
        y_pred = self.encode_y(x)
        losses = []
        N = x.shape[0]
        for i in range(K):
            y = F.one_hot(torch.full(N, i))
            elbo = self.elbo_known_y(x, y, alpha=0, lbd=lbd, mc_samples=mc_samples)
            losses.append(elbo)
        losses = torch.cat(losses, dim=1)
        loss = -torch.sum(y_pred * losses, dim=1)
        entropy = torch.sum(y_pred * torch.log(y_pred), dim=1)
        return torch.mean(loss + entropy)

    def save_to_drive(self, name=MODEL_DEFAULT_NAME):
        torch.save(self.state_dict(), os.path.join(MODELS_DIR, name+MODELS_EXT))

    @staticmethod
    def load_from_drive(K, hidden, decoder_dims, name=MODEL_DEFAULT_NAME):
        model = M2(K, hidden, decoder_dims)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT)))
        model.eval()
        return model
