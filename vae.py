import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib

from datasets import PC_OUT_DIM
from modelutils import SimplePointnetEncoder, SaveableModule, prep_seq, cd

ALPHA_PRIOR = 0.01
BETA_PRIOR = 0.01

def logbeta(a, b):
    return torch.mvlgamma(a,1)+torch.mvlgamma(b,1)-torch.mvlgamma(a+b,1)

def normal_kl(z_mean, z_log_sigma2, lbd=0.0):
    return torch.mean(torch.clamp(
        0.5*torch.sum(torch.exp(z_log_sigma2) + z_mean**2 - z_log_sigma2 - 1.0, dim=1), min=lbd))

def general_beta_kl(alpha1, beta1, alpha2, beta2, lbd=0.0):
    return torch.mean(torch.clamp(
        torch.sum(
            logbeta(alpha2, beta2)-logbeta(alpha1, beta1)
            + (alpha1-alpha2)*torch.digamma(alpha1)
            + (beta1-beta2)*torch.digamma(beta1)
            + (alpha2-alpha1+beta2-beta1)*torch.digamma(alpha1+beta1)
        , dim=1), min=lbd))

def beta_kl(log_alpha, log_beta, lbd=0.0):
    return general_beta_kl(torch.exp(log_alpha), torch.exp(log_beta),
        torch.full_like(log_alpha, ALPHA_PRIOR), torch.full_like(log_beta, BETA_PRIOR), lbd=lbd)

def normal_sample(z_mean, z_log_sigma2):
    return distrib.normal.Normal(z_mean, torch.exp(0.5*z_log_sigma2)).rsample()
    # z_sigma = torch.exp(0.5 * z_log_sigma2)
    # epsilon = torch.randn_like(z_mean)
    # return epsilon*z_sigma + z_mean

def beta_sample(log_alpha, log_beta):
    # return distrib.beta.Beta(torch.exp(alpha), torch.exp(beta)).rsample()
    x = distrib.gamma.Gamma(torch.exp(log_alpha), torch.ones_like(log_alpha)).rsample()
    y = distrib.gamma.Gamma(torch.exp(log_beta), torch.ones_like(log_beta)).rsample()
    return x/(x+y)


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
            self.kl_loss = beta_kl
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
