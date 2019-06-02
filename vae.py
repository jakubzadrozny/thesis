import torch
from torch import nn
import torch.nn.functional as F

from modelutils import SimplePointnetEncoder, SaveableModule, prep_seq, cd, gaussian_sample

LATENT = 128
ENCODER_HIDDEN = 2048
OUT_DIM = 3*2048
DECODER_DIMS = [LATENT, 1024, ENCODER_HIDDEN, ENCODER_HIDDEN, OUT_DIM]

MNIST_LATENT = 20
MNIST_HIDDEN = 500
MNIST_INPUT_DIM = 784


class VAE(SaveableModule):

    DEFAULT_SAVED_NAME = 'vae'

    def __init__(self):
        super().__init__()
        self.mean_encoder = SimplePointnetEncoder(ENCODER_HIDDEN, LATENT)
        self.sigma_encoder = SimplePointnetEncoder(ENCODER_HIDDEN, LATENT)
        self.decoder = nn.Sequential(
            prep_seq(*DECODER_DIMS),
            nn.Sigmoid(),
        )

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
        x = (x + 1.0) / 2
        return torch.mean(cd(rec, x))

    def elbo_loss(self, x, M=1, lbd=0.0):
        z_mean, z_log_sigma2 = self.encode(x)
        KL_loss = torch.mean(torch.clamp(
            -0.5*torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp(), dim=1),
            min=lbd
        ))

        rec_loss = 0
        for i in range(M):
            rec = self.decode(z_mean, z_log_sigma2, x.shape)
            rec_loss += self.rec_loss(x, rec)
        rec_loss /= M

        return (rec_loss + KL_loss,
                {'rec': rec_loss.item(), 'KL': KL_loss.item()})


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
