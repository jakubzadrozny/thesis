import os.path

import torch
from torch import nn
import torch.nn.functional as F

from pointnet import PointNetfeat
from datasets import one_hot

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

def one_hot(y, K):
    N = y.shape[0]
    x = torch.zeros((N, K)).to(y.device)
    x[torch.arange(0, N, 1), y] = 1
    return x

def gaussian_sample(z_mean, z_log_sigma2):
    z_sigma = torch.exp(0.5 * z_log_sigma2)
    epsilon = torch.randn_like(z_mean)
    return epsilon*z_sigma + z_mean


class SimplePointnetEncoder(nn.Module):
    def __init__(self, hidden, latent):
        super().__init__()
        self.model = PointNetfeat()
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden, latent)
        self.bnorm1 = nn.BatchNorm1d(1024)
        self.bnorm2 = nn.BatchNorm1d(hidden)

    def forward(self, x):
        x = F.relu(self.model(x)[0])
        x = self.bnorm1(x)
        x = F.relu(self.fc1(x))
        x = self.bnorm2(x)
        x = self.fc2(x)
        return x


class PointnetSoftmaxEncoder(SimplePointnetEncoder):
    def forward(self, x):
        x = super().forward(x)
        return F.log_softmax(x, dim=1)


class ClassPointentEncoder(nn.Module):
    def __init__(self, hidden, K, latent):
        super().__init__()
        self.feats = PointNetfeat()
        self.fc1 = nn.Linear(1024 + K, hidden)
        self.fc2 = nn.Linear(hidden, latent)
        self.bnorm1 = nn.BatchNorm1d(1024)
        self.bnorm2 = nn.BatchNorm1d(hidden)

    def forward(self, x, y):
        x = F.relu(self.feats(x)[0])
        x = self.bnorm1(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.fc1(x))
        x = self.bnorm2(x)
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


class ClassMLPDecoder(nn.Module):
    def __init__(self, K, dims):
        super().__init__()
        self.y_transform = nn.Linear(K, dims[1])
        self.z_transform = nn.Linear(dims[0], dims[1])
        self.tail = MLPDecoder(dims[1:])

    def forward(self, y, z):
        y = self.y_transform(y)
        z = self.z_transform(z)
        h = F.relu(y + z)
        return self.tail(h)


class ModifiedMLPDecoder(nn.Module):
    def __init__(self, K, dims):
        self.y_to_mean = nn.Linear(K, dims[0])
        self.y_to_cov = nn.Linear(K, dims[0])
        self.tail = MLPDecoder(dims)

    def forward(self, y, z):
        mean = self.y_to_mean(y)
        cov_diag = torch.sqrt(F.softplus(self.y_to_cov(y)))
        cov_mat = torch.diag(cov_diag)
        noise = torch.matmul(z, cov_mat)
        h = F.relu(mean + noise)
        return self.tail(h)

# class SoftmaxClassMLPDecoder(ClassMLPDecoder):
#     def forward(self, x, y):
#         x = super().forward(x, y)
#         return F.softmax(x, dim=1)
#
#
# class MLPSoftmaxEncoder(MLPDecoder):
#     def forward(self, x):
#         x = super().forward(x)
#         return F.log_softmax(x, dim=1)


class VAE(nn.Module):
    def __init__(self, hidden, decoder_dims):
        super().__init__()
        latent = decoder_dims[0]
        self.mean_encoder = SimplePointnetEncoder(hidden, latent)
        self.sigma_encoder = SimplePointnetEncoder(hidden, latent)
        self.decoder = MLPDecoder(decoder_dims)

    def encode(self, x):
        return self.mean_encoder(x), self.sigma_encoder(x)

    def decode(self, z_mean, z_log_sigma2, shape):
        z = gaussian_sample(z_mean, z_log_sigma2)
        rec = self.decoder(z)
        return rec.view(shape)

    def forward(self, x):
        z_mean, z_log_sigma2 = self.encode(x)
        rec = self.decode(z_mean, z_log_sigma2, x.shape)
        return rec, z_mean, z_log_sigma2

    def elbo_loss(self, x, mc_samples=1, lbd=0.0):
        z_mean, z_log_sigma2 = self.encode(x)
        KL_loss = torch.mean(torch.clamp(
            -0.5*torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp(), dim=1),
            min=lbd
        ))

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
        self.num_classes = K
        # self.class_encoder = MLPSoftmaxEncoder([784, 2048, 2048, K])
        # self.sigma_encoder = MLPDecoder([784, 2048, 2048, 50])
        # self.mean_encoder = ClassMLPDecoder(K, [784, 2048, 2048, 50])
        self.class_encoder = PointnetSoftmaxEncoder(hidden, K)
        self.sigma_encoder = SimplePointnetEncoder(hidden, latent)
        self.mean_encoder = ClassPointentEncoder(hidden, K, latent)
        self.decoder = ClassMLPDecoder(K, DECODER_LAYERS)

    def encode_y(self, x):
        return self.class_encoder(x)

    def encode_z_based_on_y(self, x, y):
        z_log_sigma2 = self.sigma_encoder(x)
        z_mean = self.mean_encoder(x, y)
        return z_mean, z_log_sigma2

    def decode(self, z_mean, z_log_sigma2, y, shape):
        z = gaussian_sample(z_mean, z_log_sigma2)
        rec = self.decoder(y, z)
        return rec.view(shape)

    def vectorized_elbo_known_y(self, x, y, lbd=0.0, mc_samples=1):
        z_mean, z_log_sigma2 = self.encode_z_based_on_y(x, y)
        KL_loss = torch.clamp(
            -0.5*torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp(), dim=1),
            min=lbd
        )
        # y_penalty = -torch.log(torch.tensor(1/self.num_classes))
        rec_loss = []
        for i in range(mc_samples):
            rec = self.decode(z_mean, z_log_sigma2, y, x.shape)
            rec_loss.append(cd(x, rec))
            # rec_loss.append(torch.mean(F.binary_cross_entropy(rec, x, reduction='none'), dim=1))
        rec_loss = torch.stack(rec_loss, dim=1)
        rec_loss = torch.mean(rec_loss, dim=1)
        return rec_loss, KL_loss

    def elbo_known_y(self, x, y, alpha=1.0, lbd=0.0, mc_samples=1, epsilon=1e-6):
        rec_loss, KL_loss = self.vectorized_elbo_known_y(x, y, lbd=lbd, mc_samples=mc_samples)
        rec_loss = torch.mean(rec_loss)
        KL_loss = torch.mean(KL_loss)
        log_prob_y = self.encode_y(x)
        clf_loss = -alpha*torch.mean(torch.sum(log_prob_y * y, dim=1))
        return rec_loss + KL_loss + clf_loss, {'rec': rec_loss.item(), 'KL': KL_loss.item(), 'clf': clf_loss.item()}

    def elbo_unknown_y(self, x, lbd=0.0, mc_samples=1):
        log_prob_y = self.encode_y(x)
        prob_y = torch.exp(log_prob_y)
        losses = []
        N = x.shape[0]
        for i in range(self.num_classes):
            y = one_hot(torch.full((N,), i, dtype=torch.long, device=x.device), self.num_classes)
            rec_loss, KL_loss = self.vectorized_elbo_known_y(x, y, lbd=lbd, mc_samples=mc_samples)
            losses.append(rec_loss + KL_loss)
        loss_guessed_y = torch.stack(losses, dim=1)
        loss = torch.mean(torch.sum(prob_y * loss_guessed_y, dim=1))
        entropy = -torch.mean(torch.sum(prob_y * log_prob_y, dim=1))
        return loss + entropy, {'loss': loss.item(), 'entropy': entropy.item()}

    def save_to_drive(self, name=MODEL_DEFAULT_NAME):
        torch.save(self.state_dict(), os.path.join(MODELS_DIR, name+MODELS_EXT))

    @staticmethod
    def load_from_drive(K, hidden, decoder_dims, name=MODEL_DEFAULT_NAME):
        model = M2(K, hidden, decoder_dims)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT)))
        model.eval()
        return model


class ModifiedM2(M2):
    def __init__(K, hidden, decoder_dims):
        super().__init__(K, hidden, decoder_dims)
        self.decoder = ModifiedMLPDecoder(K, DECODER_LAYERS)
