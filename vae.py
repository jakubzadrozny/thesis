import torch
from torch import nn
import torch.nn.functional as F

from modelutils import SimplePointnetEncoder, SaveableModule, prep_seq, cd, gaussian_sample

LATENT = 128
HIDDEN = 1024
OUT_DIM = 3*2048
ENCODER_DIMS = [OUT_DIM, HIDDEN, HIDDEN, HIDDEN, 2*LATENT]
DECODER_DIMS = [LATENT, HIDDEN, HIDDEN, HIDDEN, OUT_DIM]

MNIST_LATENT = 20
MNIST_HIDDEN = 500
MNIST_INPUT_DIM = 784


class VAE(SaveableModule):

    def __init__(self):
        super().__init__()

    def encode(self, x):
        x = self.encoder(x)
        return torch.chunk(x, 2, dim=1)

    def decode(self, z_mean, z_log_sigma2):
        z = gaussian_sample(z_mean, z_log_sigma2)
        rec = self.decoder(z)
        return rec

    def forward(self, x):
        z_mean, z_log_sigma2 = self.encode(x)
        rec = self.decode(z_mean, z_log_sigma2).view(x.shape)
        return rec, z_mean, z_log_sigma2

    def elbo_loss(self, x, M=1, lbd=0.0):
        z_mean, z_log_sigma2 = self.encode(x)
        KL_loss = torch.mean(torch.clamp(
            -0.5*torch.sum(1.0 + z_log_sigma2 - z_mean**2- torch.exp(z_log_sigma2), dim=1),
            min=lbd
        ))

        rec_loss = 0
        for i in range(M):
            rec = self.decode(z_mean, z_log_sigma2).view(x.shape)
            rec_loss += self.rec_loss(x, rec)
        rec_loss /= M

        return (rec_loss + KL_loss,
                {'rec': rec_loss.item(), 'KL': KL_loss.item()})


class PCVAE(VAE):

    DEFAULT_SAVED_NAME = 'pcvae'

    def __init__(self, outvar=2e-3):
        super(VAE, self).__init__()
        self.outvar = outvar
        self.decoder = prep_seq(*DECODER_DIMS)
        # self.encoder = prep_seq(*ENCODER_DIMS, bnorm=True)
        self.encoder = SimplePointnetEncoder(HIDDEN, 2*LATENT)
        # self.sigma_encoder = SimplePointnetEncoder(ENCODER_HIDDEN, LATENT)
        # self.mean_encoder = SimplePointnetEncoder(ENCODER_HIDDEN, 2*LATENT)

    # def encode(self, x):
    #     x = x.reshape(x.shape[0], -1)
    #     return super().encode(x)

    def rec_loss(self, x, rec):
        return 1/(2*self.outvar) * torch.mean(cd(rec, x))


class MNISTVAE(VAE):

    DEFAULT_SAVED_NAME = 'mnistvae'

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
