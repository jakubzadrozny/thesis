import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib

from datasets import PC_OUT_DIM
from modelutils import (SimplePointnetEncoder, ExpPointnetEncoder, SaveableModule,
                        prep_seq, cd, logbeta)


class AE(SaveableModule):
    DEFAULT_SAVED_NAME = 'ae'

    def __init__(self, latent=128, decoder=[512, 1024, 1024, 2048], encoder=[]):
        super().__init__()
        self.decoder = prep_seq(latent, *decoder, PC_OUT_DIM)
        self.encoder = SimplePointnetEncoder(*encoder, latent)

    def rec_loss(self, x, rec):
        return torch.mean(cd(rec, x))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z).view(x.shape), z

    def elbo_loss(self, x, M=1):
        z = self.encoder(x)
        rec = self.decoder(z).view(x.shape)
        return self.rec_loss(x, rec)


class VAE(SaveableModule):

    def __init__(self):
        super().__init__()

    def encode(self, x):
        x = self.encoder(x)
        alpha, beta = torch.chunk(x, 2, dim=1)
        z = self.sample(alpha, beta)
        return z, alpha, beta

    def forward(self, x, **kwargs):
        z, _, _ = self.encode(x, **kwargs)
        return self.decoder(z).view(x.shape), z

    def elbo_loss(self, x, M=1):
        _, alpha, beta = self.encode(x)
        KL_loss = self.kl_loss(alpha, beta)

        rec_loss = 0
        for i in range(M):
            z = self.sample(alpha, beta)
            rec = self.decoder(z).view(x.shape)
            rec_loss += self.rec_loss(x, rec)
        rec_loss /= M

        return (rec_loss + KL_loss,
                {'rec': rec_loss.item(), 'KL': KL_loss.item()})


class NPCVAE(VAE):
    DEFAULT_SAVED_NAME = 'normal_pcvae'

    def __init__(self, latent=128, decoder=[512, 1024, 1024, 2048], encoder=[], rec_var=1e-3, latent_var=1.0):
        super().__init__()
        self.rec_var_inv = 1.0/rec_var
        self.latent_var = latent_var
        self.latent_var_inv = 1.0 / self.latent_var
        self.latent_var_log = torch.log(torch.tensor(self.latent_var)).item()

        self.decoder = prep_seq(latent, *decoder, PC_OUT_DIM)
        self.encoder = SimplePointnetEncoder(*encoder, 2*latent)

    def sample(self, z_mean, z_log_sigma2):
        z_sigma = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_mean)
        return epsilon*z_sigma + z_mean

    def rec_loss(self, x, rec):
        return self.rec_var_inv*torch.mean(cd(rec, x))

    def kl_loss(self, z_mean, z_log_sigma2):
        return torch.mean(0.5*torch.sum(
                    self.latent_var_inv*(torch.exp(z_log_sigma2) + z_mean**2)
                    - z_log_sigma2 - 1.0 + self.latent_var_log, dim=1))


class BPCVAE(VAE):
    DEFAULT_SAVED_NAME = 'beta_pcvae'

    def __init__(self, latent=128, decoder=[512, 1024, 1024, 2048], encoder=[], rec_var=1e-3,
                 alpha_prior=0.01, beta_prior=0.01):
        super().__init__()
        self.rec_var_inv = 1.0/rec_var
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.prior_logbeta = logbeta(torch.tensor(alpha_prior), torch.tensor(beta_prior))

        self.decoder = prep_seq(latent, *decoder, PC_OUT_DIM)
        self.encoder = ExpPointnetEncoder(*encoder, 2*latent)

    def encode(self, x, binarize=False):
        z, alpha, beta = super().encode(x)
        if binarize:
            z = (z > 0.5).float()
        return z, alpha, beta

    def sample(self, alpha, beta, eps=1e-6):
        # return distrib.beta.Beta(torch.exp(alpha), torch.exp(beta)).rsample()
        x = distrib.gamma.Gamma(alpha, torch.ones_like(alpha)).rsample()
        y = distrib.gamma.Gamma(beta, torch.ones_like(beta)).rsample()
        return torch.clamp(x/(x+y+eps), min=eps, max=1-eps)

    def rec_loss(self, x, rec):
        return self.rec_var_inv*torch.mean(cd(rec, x))

    def kl_loss(self, alpha, beta):
        return torch.mean(torch.sum(
                self.prior_logbeta-logbeta(alpha, beta)
                + (alpha-self.alpha_prior)*torch.digamma(alpha)
                + (beta-self.beta_prior)*torch.digamma(beta)
                + (self.alpha_prior-alpha+self.beta_prior-beta)*torch.digamma(alpha+beta)
                , dim=1))

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
