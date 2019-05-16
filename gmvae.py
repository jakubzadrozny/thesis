import torch
from torch import nn
import torch.nn.functional as F

from modelutils import (SimplePointnetEncoder, PointnetSoftmaxEncoder, ClassPointentEncoder,
                        ClassMLP, SigmoidClassMLP, GMDecoder, SigmoidGMDecoder, SaveableModule,
                        prep_seq, cd, gaussian_sample, one_hot)

LATENT = 128
ENCODER_HIDDEN = 1024
OUT_DIM = 3*2048
DECODER_DIMS = [LATENT, 512, 1024, 1024, 2048, OUT_DIM]
NUM_CLASSESS = 3

MNIST_LATENT = 20
MNIST_HIDDEN = 500
MNIST_INPUT_DIM = 784
MNIST_CLASSES = 10


class GMVAE(SaveableModule):

    DEFAULT_SAVED_NAME = 'gmvae'

    def __init__(self, K):
        super().__init__(K=NUM_CLASSESS)
        self.num_classes = K
        self.class_encoder = PointnetSoftmaxEncoder(ENCODER_HIDDEN, self.num_classes)
        self.sigma_encoder = SimplePointnetEncoder(ENCODER_HIDDEN, LATENT)
        self.mean_encoder = ClassPointentEncoder(ENCODER_HIDDEN, self.num_classes, LATENT)
        self.decoder = prep_seq(*DECODER_DIMS)
        self.gm_components = prep_seq(self.num_classes, 2*LATENT, 2*LATENT)

    def encode_y(self, x):
        return self.class_encoder(x)

    def get_gm_components(self, y):
        components = self.gm_components(y)
        return torch.chunk(components, 2, dim=1)

    def encode_z_based_on_y(self, x, y):
        z_log_sigma2 = self.sigma_encoder(x)
        z_mean = self.mean_encoder(x, y)
        return z_mean, z_log_sigma2

    def decode(self, z_mean, z_log_sigma2, shape):
        z = gaussian_sample(z_mean, z_log_sigma2)
        rec = self.decoder(z)
        return rec.view(shape)

    def rec_loss(self, x, rec):
        return cd(x, rec)

    def elbo_known_y(self, x, y, lbd=0.0, M=1):
        z_mean, z_log_sigma2 = self.encode_z_based_on_y(x, y)
        components_mean, components_log_sigma2 = self.get_gm_components(y)
        mean_diff = components_mean - z_mean
        KL_loss = torch.clamp(
            0.5*torch.sum(
                -1.0 +
                torch.exp(z_log_sigma2-components_log_sigma2) +
                components_log_sigma2 - z_log_sigma2 +
                (mean_diff ** 2) * torch.exp(-components_log_sigma2),
                dim=1),
            min=lbd
        )
        # y_penalty = -torch.log(torch.tensor(1/self.num_classes)).to(y.device)
        rec_loss = []
        for i in range(M):
            rec = self.decode(z_mean, z_log_sigma2, x.shape)
            rec_loss.append(self.rec_loss(x, rec))
        rec_loss = torch.mean(torch.stack(rec_loss, dim=1), dim=1)
        return rec_loss, KL_loss #, y_penalty

    def elbo_loss(self, x, lbd=0.0, M=1):
        log_prob_y = self.encode_y(x)
        prob_y = torch.exp(log_prob_y)
        losses = []
        N = x.shape[0]
        for i in range(self.num_classes):
            y = one_hot(torch.full((N,), i, dtype=torch.long, device=x.device), self.num_classes)
            rec_loss, KL_loss = self.elbo_known_y(x, y, lbd=lbd, M=M)
            losses.append(rec_loss + KL_loss)
        loss_guessed_y = torch.stack(losses, dim=1)
        loss = torch.mean(torch.sum(prob_y * loss_guessed_y, dim=1))
        entropy = torch.mean(torch.sum(prob_y * log_prob_y, dim=1))
        return loss + entropy, {'loss': loss.item(), 'entropy': entropy.item()}

    def classify(self, x):
        y = self.encode_y(x)
        return torch.argmax(y, dim=1)


class MNISTGMVAE(GMVAE):

    DEFAULT_SAVED_NAME = 'mnistmgmvae'

    def __init__(self):
        super(SaveableModule, self).__init__()
        self.num_classes = MNIST_CLASSES
        self.class_encoder = nn.Sequential(
            prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, self.num_classes),
            nn.LogSoftmax(dim=1),
        )
        self.sigma_encoder = prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
        self.mean_encoder = ClassMLP(self.num_classes,
                                     MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_LATENT)
        self.decoder = nn.Sequential(
            prep_seq(MNIST_LATENT, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_INPUT_DIM),
            nn.Sigmoid(),
        )
        self.gm_components = prep_seq(self.num_classes, 2*MNIST_LATENT, 2*MNIST_LATENT)

    def rec_loss(self, x, rec):
        return torch.sum(F.binary_cross_entropy(rec, x, reduction='none'), dim=1)
