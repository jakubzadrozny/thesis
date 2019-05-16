import os.path

import torch
from torch import nn
import torch.nn.functional as F

from pointnet import PointNetfeat

MODELS_DIR = 'trained'
MODELS_EXT = '.dms'


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


def prep_seq(*dims):
    layers = []
    for i in range(len(dims)-2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


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


class GMDecoder(nn.Module):
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


class SigmoidGMDecoder(GMDecoder):
    def forward(self, y, z):
        x = super().forward(y, z)
        return torch.sigmoid(x)


class SaveableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save_to_drive(self, name=None):
        name = name if name is not None else self.DEFAULT_SAVED_NAME
        torch.save(self.state_dict(), os.path.join(MODELS_DIR, name+MODELS_EXT))

    @staticmethod
    def load_from_drive(model, name=None):
        name = name if name is not None else model.DEFAULT_SAVED_NAME
        loaded = model()
        loaded.load_state_dict(torch.load(os.path.join(MODELS_DIR, name+MODELS_EXT)))
        loaded.eval()
        return loaded
