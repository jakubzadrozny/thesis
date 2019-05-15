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

MNIST_LATENT = 20
MNIST_HIDDEN = 500
MNIST_INPUT_DIM = 784

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


def prep_seq(*dims):
    layers = []
    for i in range(len(dims)-2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class ClassMLP(nn.Module):
    def __init__(self, K, *layers):
        super().__init__()
        self.tail = prep_seq(layers[0]+K, *layers[1:])

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        return self.tail(x)


class SigmoidClassMLP(ClassMLP):
    def forward(self, x, y):
        x = super().forward(x, y)
        return torch.sigmoid(x)


class ModifiedMLP(nn.Module):
    def __init__(self, K, dims):
        self.y_to_mean = nn.Linear(K, dims[0])
        self.y_to_cov = nn.Linear(K, dims[0])
        self.tail = prep_seq(*dims)

    def forward(self, y, z):
        mean = self.y_to_mean(y)
        cov_diag = torch.sqrt(F.softplus(self.y_to_cov(y)))
        cov_mat = torch.diag(cov_diag)
        noise = torch.matmul(z, cov_mat)
        h = F.relu(mean + noise)
        return self.tail(h)


class VAE(nn.Module):
    def __init__(self, hidden, decoder_dims):
        super().__init__()
        latent = decoder_dims[0]
        self.mean_encoder = SimplePointnetEncoder(hidden, latent)
        self.sigma_encoder = SimplePointnetEncoder(hidden, latent)
        self.decoder = prep_seq(*decoder_dims)

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

    def rec_loss(self, x, rec):
        return torch.mean(cd(x, rec))

    def elbo_loss(self, x, mc_samples=1, lbd=0.0):
        z_mean, z_log_sigma2 = self.encode(x)
        KL_loss = torch.mean(torch.clamp(
            -0.5*torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp(), dim=1),
            min=lbd
        ))

        rec_loss = 0
        for i in range(mc_samples):
            rec = self.decode(z_mean, z_log_sigma2, x.shape)
            rec_loss += self.rec_loss(x, rec)
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


class MNISTVAE(VAE):
    def __init__(self):
        super(VAE, self).__init__()
        self.mean_encoder = prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
        self.sigma_encoder = prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
        self.decoder = nn.Sequential(
            prep_seq(MNIST_LATENT, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_INPUT_DIM),
            nn.Sigmoid(),
        )

    def rec_loss(self, x, rec):
        return F.binary_cross_entropy(rec, x, reduction='sum') / x.shape[0]

    @staticmethod
    def load_from_drive(name=MODEL_DEFAULT_NAME):
        model = MNISTVAE()
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT)))
        model.eval()
        return model


class M2(nn.Module):
    def __init__(self, K, hidden, decoder_dims):
        super().__init__()
        latent = decoder_dims[0]
        self.num_classes = K
        self.class_encoder = PointnetSoftmaxEncoder(hidden, K)
        self.sigma_encoder = SimplePointnetEncoder(hidden, latent)
        self.mean_encoder = ClassPointentEncoder(hidden, K, latent)
        self.decoder = ClassMLP(K, decoder_dims)

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

    def rec_loss(self, x, rec):
        return cd(x, rec)

    def vectorized_elbo_known_y(self, x, y, lbd=0.0, mc_samples=1):
        z_mean, z_log_sigma2 = self.encode_z_based_on_y(x, y)
        KL_loss = torch.clamp(
            -0.5*torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp(), dim=1),
            min=lbd
        )
        y_penalty = -torch.log(torch.tensor(1/self.num_classes))
        rec_loss = []
        for i in range(mc_samples):
            rec = self.decode(z_mean, z_log_sigma2, y, x.shape)
            rec_loss.append(self.rec_loss(x, rec))
        rec_loss = torch.stack(rec_loss, dim=1)
        rec_loss = torch.mean(rec_loss, dim=1)
        return rec_loss, KL_loss, y_penalty

    def elbo_known_y(self, x, y, alpha=1.0, lbd=0.0, mc_samples=1, epsilon=1e-6):
        rec_loss, KL_loss, y_penalty = self.vectorized_elbo_known_y(x, y, lbd=lbd, mc_samples=mc_samples)
        rec_loss = torch.mean(rec_loss)
        KL_loss = torch.mean(KL_loss)
        y_penalty = torch.mean(y_penalty)
        log_prob_y = self.encode_y(x)
        clf_loss = -alpha*torch.mean(torch.sum(log_prob_y * y, dim=1))
        return rec_loss + KL_loss + y_penalty + clf_loss, {'rec': rec_loss.item(), 'KL': KL_loss.item(), 'clf': clf_loss.item(), 'y_pen': y_penalty.item()}

    def elbo_unknown_y(self, x, lbd=0.0, mc_samples=1):
        log_prob_y = self.encode_y(x)
        prob_y = torch.exp(log_prob_y)
        losses = []
        N = x.shape[0]
        for i in range(self.num_classes):
            y = one_hot(torch.full((N,), i, dtype=torch.long, device=x.device), self.num_classes)
            rec_loss, KL_loss, y_penalty = self.vectorized_elbo_known_y(x, y, lbd=lbd, mc_samples=mc_samples)
            losses.append(rec_loss + KL_loss + y_penalty)
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


class MNISTM2(M2):
    def __init__(self):
        super(M2, self).__init__()
        self.num_classes = 10
        self.class_encoder = nn.Sequential(
            prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, self.num_classes),
            nn.LogSoftmax(dim=1),
        )
        self.sigma_encoder = prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
        self.mean_encoder = ClassMLP(self.num_classes,
                                     MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
        self.decoder = SigmoidClassMLP(self.num_classes,
                                       MNIST_LATENT, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_INPUT_DIM)

    def rec_loss(self, x, rec):
        return torch.sum(F.binary_cross_entropy(rec, x, reduction='none'), dim=1)

    @staticmethod
    def load_from_drive(name=MODEL_DEFAULT_NAME):
        model = MNISTM2()
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT)))
        model.eval()
        return model


class ModifiedM2(M2):
    def __init__(K, hidden, decoder_dims):
        super().__init__(K, hidden, decoder_dims)
        self.decoder = ModifiedMLPDecoder(K, DECODER_LAYERS)
