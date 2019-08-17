import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib

from datasets import PC_OUT_DIM
from modelutils import SimplePointnetEncoder, SaveableModule, prep_seq, cd

ALPHA_PRIOR = 0.01
BETA_PRIOR = 0.01

def exp_with_eps(x, eps=1e-6):
    return torch.clamp(torch.exp(x), min=eps, max=1.0/eps)

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

def beta_kl(alpha, beta, lbd=0.0):
    return general_beta_kl(alpha, beta,
                           torch.full_like(alpha, ALPHA_PRIOR), torch.full_like(beta, BETA_PRIOR), lbd=lbd)

def normal_sample(z_mean, z_log_sigma2):
    z_sigma = torch.exp(0.5 * z_log_sigma2)
    epsilon = torch.randn_like(z_mean)
    return epsilon*z_sigma + z_mean

def beta_sample(alpha, beta, eps=1e-6):
    # return distrib.beta.Beta(torch.exp(alpha), torch.exp(beta)).rsample()
    x = distrib.gamma.Gamma(alpha, torch.ones_like(alpha)).rsample()
    y = distrib.gamma.Gamma(beta, torch.ones_like(beta)).rsample()
    return torch.clamp(x/(x+y+eps), min=eps, max=1-eps)


class VAE(SaveableModule):

    def __init__(self):
        super().__init__()
        self.postprocess = lambda x: x
        # self.latent_var = latent_var
        # self.latent_var_inv = 1.0 / self.latent_var
        # self.latent_var_log = torch.log(torch.tensor(self.latent_var)).item()

    def encode(self, x):
        x = self.encoder(x)
        x = self.postprocess(x)
        return torch.chunk(x, 2, dim=1)

    def encode_sample(self, x):
        alpha, beta = self.encode(x)
        return self.sample(alpha, beta)

    def encode_sample_b(self, x):
        z = self.encode_sample(x)
        return (z > 0.5).float()

    def decode(self, alpha, beta):
        z = self.sample(alpha, beta)
        rec = self.decoder(z)
        return rec, z

    def decode_at_mean(self, z_mean):
        return self.decoder(z_mean)

    def forward(self, x):
        alpha, beta = self.encode(x)
        rec, z = self.decode(alpha, beta)
        return rec.view(x.shape), z

    def decode_binarized(self, alpha, beta):
        z = (self.sample(alpha, beta) > 0.5).float()
        rec = self.decoder(z)
        return rec, z

    def forward_binarized(self, x):
        alpha, beta = self.encode(x)
        rec, z = self.decode_binarized(alpha, beta)
        return rec.view(x.shape), z

    def elbo_loss(self, x, M=1, lbd=0.0):
        alpha, beta = self.encode(x)
        KL_loss = self.kl_loss(alpha, beta, lbd=lbd)

        rec_loss = 0
        for i in range(M):
            rec, _ = self.decode(alpha, beta)
            rec = rec.view(x.shape)
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
            self.postprocess = exp_with_eps

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
