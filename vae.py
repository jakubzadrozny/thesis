import numpy as np

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.distributions as distrib

from datasets import PC_OUT_DIM
from modelutils import (SimplePointnetEncoder, SaveableModule,
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
        recl = self.rec_loss(x, rec)
        return recl, {'rec': recl.item()}


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
        self.encoder = SimplePointnetEncoder(*encoder, 2*latent)

    def encode(self, x, binarize=False):
        x = torch.exp(self.encoder(x))
        alpha, beta = torch.chunk(x, 2, dim=1)
        z = self.sample(alpha, beta)
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


# class GMSample(Function):
#
#     @staticmethod
#     def forward(ctx, logits, components):
#         # print(logits)
#         idx = distrib.Categorical(logits=logits).sample().view(-1, 1, 1).expand(-1, -1, components.shape[2])
#         params = torch.gather(components, 1, idx).squeeze(1)
#         z_mean, z_log_sigma2 = torch.chunk(params, 2, dim=1)
#         z_sigma = torch.exp(0.5*z_log_sigma2)
#         z = torch.normal(z_mean, z_sigma)
#         ctx.save_for_backward(logits, components, z)
#         return z
#
#     @staticmethod
#     def backward(ctx, grad_z):
#         logits, components, z = ctx.saved_tensors
#         grad_logits = grad_components = None
#
#         if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
#             return grad_logits, grad_components
#
#         weights = F.softmax(logits, dim=1)
#         weights_exp = weights.unsqueeze(2)
#         z_mean, z_log_sigma2 = torch.chunk(components, 2, dim=2)
#         z_sigma = torch.exp(0.5*z_log_sigma2)
#         d = distrib.normal.Normal(z_mean, z_sigma)
#         expz = z.unsqueeze(1).expand(-1, components.shape[1], -1)
#         probs = torch.exp(d.log_prob(expz))
#         w_probs = probs * weights_exp
#         total_probs = torch.sum(w_probs, dim=1, keepdim=True)
#
#         dz_dmean = w_probs / total_probs
#         dz_dlog_sigma2 = w_probs * (expz - z_mean) /  2*total_probs
#
#         exp_grad_z = grad_z.unsqueeze(1)
#         grad_mean = dz_dmean * exp_grad_z
#         grad_log_sigma2 = dz_dlog_sigma2 * exp_grad_z
#
#         grad_components = torch.cat((grad_mean, grad_log_sigma2), dim=2)
#
#         dcdfs_dweights = d.cdf(expz)
#         dweights_dlogits = (-weights_exp * weights.unsqueeze(1) +
#                             weights_exp * torch.eye(weights.shape[1], device=weights.device).unsqueeze(0))
#         dcdfs_dlogits = torch.matmul(dweights_dlogits, dcdfs_dweights)
#         dz_dlogits = -dcdfs_dlogits / total_probs
#
#         grad_logits = torch.matmul(dz_dlogits, grad_z.unsqueeze(2)).squeeze(2)
#
#         return grad_logits, grad_components
#
#
# class GPCVAE(VAE):
#     DEFAULT_SAVED_NAME = 'gm_pcvae'
#
#     def __init__(self, clusters, prior_components,
#                  latent=128, decoder=[512, 1024, 1024, 2048], encoder=[], rec_var=1e-3):
#         super().__init__()
#         self.rec_var_inv = 1.0/rec_var
#         self.latent = latent
#         self.clusters = clusters
#         self.log_clusters = torch.log(torch.tensor(clusters, dtype=torch.float)).item()
#         self.log_2pi = torch.log(torch.tensor(2*np.pi)).item()
#         self.register_buffer('prior_components', prior_components.unsqueeze(1))
#
#         self.sample = GMSample.apply
#         self.encoder = SimplePointnetEncoder(*encoder, clusters+clusters*2*latent)
#         self.decoder = prep_seq(latent, *decoder, PC_OUT_DIM)
#
#     def encode(self, x):
#         x = self.encoder(x)
#         logits, components = torch.split(x, [self.clusters, self.clusters*2*self.latent], dim=1)
#         components = components.reshape(x.shape[0], -1, 2*self.latent)
#         z = self.sample(logits, components)
#         return z, logits, components
#
#     def rec_loss(self, x, rec):
#         return self.rec_var_inv*torch.mean(cd(rec, x))
#
#     def elbo_loss(self, x, M=1):
#         _, alpha, beta = self.encode(x)
#
#         rec_loss = 0
#         for i in range(M):
#             z = self.sample(alpha, beta)
#             rec = self.decoder(z).view(x.shape)
#             rec_loss += self.rec_loss(x, rec)
#         rec_loss /= M
#
#         return (rec_loss, {'rec': rec_loss.item()})

    # def kl_loss(self, log_weights, components):
    #     weights = torch.exp(log_weights)
    #
    #     z_mean, z_log_sigma2 = torch.chunk(components, 2, dim=2)
    #     b, c, d = z_log_sigma2.shape
    #     z_log_sigma2_exp = z_log_sigma2.unsqueeze(2)
    #     z_mean_exp1 = z_mean.unsqueeze(1)
    #     z_mean_exp2 = z_mean.unsqueeze(2)
    #
    #     cross_kl = torch.sum(torch.exp(z_log_sigma2_exp) + (z_mean_exp2-self.prior_components)**2
    #                          - z_log_sigma2_exp - 1.0, dim=3)
    #     log_denom = torch.logsumexp(-cross_kl, dim=2)
    #
    #     cross_sigma2_logsum = torch.logsumexp(torch.stack((
    #         z_log_sigma2.unsqueeze(1).expand(-1, c, -1, -1),
    #         z_log_sigma2.unsqueeze(2).expand(-1, -1, c, -1),
    #     ), dim=4), dim=4)
    #     log_t = -0.5 * (d*self.log_2pi + torch.sum(cross_sigma2_logsum, dim=3) +
    #                 torch.sum(torch.exp(-cross_sigma2_logsum) * (z_mean_exp1 - z_mean_exp2)**2, dim=3))
    #
    #     log_num = torch.logsumexp(log_weights.unsqueeze(1) + log_t, dim=2)
    #
    #     first = torch.sum(weights * (log_num - log_denom - self.log_clusters), dim=1)
    #
    #     entropy_per_component = 0.5*(torch.sum(z_log_sigma2, dim=2) + d*(1+self.log_2pi))
    #     second = torch.sum(weights * entropy_per_component, dim=1)
    #
    #     return torch.mean(first + second)

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
