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


class M2(SaveableModule):

    DEFAULT_SAVED_NAME = 'm2'

    def __init__(self, K):
        super().__init__(K=NUM_CLASSESS)
        self.num_classes = K
        self.class_encoder = PointnetSoftmaxEncoder(ENCODER_HIDDEN, self.num_classes)
        self.sigma_encoder = SimplePointnetEncoder(ENCODER_HIDDEN, LATENT)
        self.mean_encoder = ClassPointentEncoder(ENCODER_HIDDEN, self.num_classes, LATENT)
        self.decoder = ClassMLP(self.num_classes, DECODER_DIMS)

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

    def vectorized_elbo_known_y(self, x, y, lbd=0.0, M=1):
        z_mean, z_log_sigma2 = self.encode_z_based_on_y(x, y)
        KL_loss = torch.clamp(
            -0.5*torch.sum(1.0 + z_log_sigma2 - z_mean.pow(2) - z_log_sigma2.exp(), dim=1),
            min=lbd
        )
        # y_penalty = -torch.log(torch.tensor(1/self.num_classes)).to(y.device)
        rec_loss = []
        for i in range(M):
            rec = self.decode(z_mean, z_log_sigma2, y, x.shape)
            rec_loss.append(self.rec_loss(x, rec))
        rec_loss = torch.mean(torch.stack(rec_loss, dim=1), dim=1)
        return rec_loss, KL_loss #, y_penalty

    def elbo_known_y(self, x, y, alpha=1.0, lbd=0.0, M=1, epsilon=1e-6):
        rec_loss, KL_loss = self.vectorized_elbo_known_y(x, y, lbd=lbd, M=M)
        rec_loss = torch.mean(rec_loss)
        KL_loss = torch.mean(KL_loss)
        # y_penalty = torch.mean(y_penalty)
        log_prob_y = self.encode_y(x)
        clf_loss = -alpha*torch.mean(torch.sum(log_prob_y * y, dim=1))
        return rec_loss + KL_loss + clf_loss, {'rec': rec_loss.item(), 'KL': KL_loss.item(), 'clf': clf_loss.item()}

    def elbo_unknown_y(self, x, lbd=0.0, M=1):
        log_prob_y = self.encode_y(x)
        prob_y = torch.exp(log_prob_y)
        losses = []
        N = x.shape[0]
        for i in range(self.num_classes):
            y = one_hot(torch.full((N,), i, dtype=torch.long, device=x.device), self.num_classes)
            rec_loss, KL_loss = self.vectorized_elbo_known_y(x, y, lbd=lbd, M=M)
            losses.append(rec_loss + KL_loss)
        loss_guessed_y = torch.stack(losses, dim=1)
        loss = torch.mean(torch.sum(prob_y * loss_guessed_y, dim=1))
        entropy = torch.mean(torch.sum(prob_y * log_prob_y, dim=1))
        return loss + entropy, {'loss': loss.item(), 'entropy': entropy.item()}

    def classify(self, x):
        y = self.encode_y(x)
        return torch.argmax(y, dim=1)


class MNISTM2(M2):

    DEFAULT_SAVED_NAME = 'mnistm2'

    def __init__(self):
        super(M2, self).__init__()
        self.num_classes = MNIST_CLASSES
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


class ModifiedM2(M2):

    DEFAULT_SAVED_NAME = 'modifiedm2'

    def __init__(K=NUM_CLASSESS):
        super().__init__(K=K)
        self.decoder = GMMLPDecoder(self.num_classes, DECODER_DIMS)

    def elbo_loss(self, x, lbd=0.0, M=1):
        return self.elbo_unknown_y(x, lbd=lbd, M=M)


class MNISTModifiedM2(M2):

    DEFAULT_SAVED_NAME = 'mnistmodifiedm2'

    def __init__():
        super().__init__()
        self.decoder = SigmoidGMMLPDecoder(self.num_classes, DECODER_DIMS)

    def elbo_loss(self, x, lbd=0.0, M=1):
        return self.elbo_unknown_y(x, lbd=lbd, M=M)
