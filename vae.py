import torch
from torch import nn
import torch.nn.functional as F

from datasets import PC_OUT_DIM
from modelutils import (SimplePointnetEncoder, SaveableModule, prep_seq, cd,
                        normal_sample, beta_sample, normal_kl, beta_kl)

ALPHA_PRIOR = 0.01
BETA_PRIOR = 0.01

def beta_kl_loss(alpha, beta, lbd=0.0):
    return kl_loss(alpha, beta, torch.full_like(alpha, ALPHA_PRIOR), torch.full_like(beta, BETA_PRIOR), lbd=lbd)

class VAE(SaveableModule):

    def __init__(self):
        super().__init__()
        # self.latent_var = latent_var
        # self.latent_var_inv = 1.0 / self.latent_var
        # self.latent_var_log = torch.log(torch.tensor(self.latent_var)).item()

    def encode(self, x):
        x = self.encoder(x)
        return torch.chunk(x, 2, dim=1)

    def decode(self, alpha, beta):
        z = self.sample(alpha, beta)
        rec = self.decoder(z)
        return rec

    def decode_at_mean(self, z_mean):
        return self.decoder(z_mean)

    def forward(self, x):
        z_mean, _ = self.encode(x)
        rec = self.decode_at_mean(z_mean).view(x.shape)
        return rec, z_mean

    def elbo_loss(self, x, M=1, lbd=0.0):
        alpha, beta = self.encode(x)
        KL_loss = self.kl_loss(alpha, beta, lbd=lbd)

        rec_loss = 0
        for i in range(M):
            rec = self.decode(alpha, beta).view(x.shape)
            rec_loss += self.rec_loss(x, rec)
        rec_loss /= M

        return (rec_loss + KL_loss,
                {'rec': rec_loss.item(), 'KL': KL_loss.item()})


class PCVAE(VAE):
    DEFAULT_SAVED_NAME = 'pcvae'

    def __init__(self, latent=128, decoder=[1024, 1024, 1024], encoder=[], rec_var=1e-3, prior='normal'):
        super().__init__()
        self.rec_var_inv = 1.0/rec_var
        self.decoder = prep_seq(latent, *decoder, PC_OUT_DIM)
        self.encoder = SimplePointnetEncoder(*encoder, 2*latent)
        if prior == 'normal':
            self.kl_loss = normal_kl
            self.sample = normal_sample
        else:
            self.kl_loss = beta_kl_loss
            self.sample = beta_sample

    def rec_loss(self, x, rec):
        return self.rec_var_inv*torch.mean(cd(rec, x))


# MNIST_LATENT = 20
# MNIST_HIDDEN = 500
# MNIST_INPUT_DIM = 784

# class MNISTVAE(VAE):
#
#     DEFAULT_SAVED_NAME = 'mnistvae'
#
#     def __init__(self):
#         super(VAE, self).__init__()
#         self.mean_encoder = prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
#         self.sigma_encoder = prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
#         self.decoder = nn.Sequential(
#             prep_seq(MNIST_LATENT, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_INPUT_DIM),
#             nn.Sigmoid(),
#         )
#
#     def rec_loss(self, x, rec):
#         return F.binary_cross_entropy(rec, x, reduction='sum') / x.shape[0]
