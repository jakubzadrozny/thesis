import torch
from torch import nn
import torch.nn.functional as F

from datasets import PC_OUT_DIM
from modelutils import (SoloPointnetEncoder, SimplePointnetEncoder,
                        SaveableModule, prep_seq, cd, one_hot, generate_random_points)


class GMVAE(SaveableModule):

    DEFAULT_SAVED_NAME = 'gmvae'

    def __init__(self, clusters, prior_means,
                 latent=128, encoder=[], decoder=[512, 1024, 1024, 2048], rec_var=1e-3):
        super().__init__()
        self.rec_var_inv = 1.0/rec_var
        self.clusters = clusters

        self.pointnet = SoloPointnetEncoder()
        self.cat_encoder = nn.Sequential(nn.Linear(1024, clusters), nn.LogSoftmax(dim=1))
        self.lat_encoder = prep_seq(1024+clusters, *encoder, 2*latent, bnorm=True)
        self.decoder = prep_seq(latent, *decoder, PC_OUT_DIM)
        self.register_buffer('means', prior_means)
        # self.components = prep_seq(clusters, 256, 2*latent)

    def sample(self, z_mean, z_log_sigma2):
        z_sigma = torch.exp(0.5 * z_log_sigma2)
        epsilon = torch.randn_like(z_mean)
        return epsilon*z_sigma + z_mean

    def encode_z(self, y, feats):
        params = self.lat_encoder(torch.cat((y, feats), dim=1))
        z_mean, z_log_sigma2 = torch.chunk(params, 2, dim=1)
        z = self.sample(z_mean, z_log_sigma2)
        return z, z_mean, z_log_sigma2

    # def get_components(self, y):
        # params = self.components(y)
        # return torch.chunk(params, 2, dim=1)

    def rec_loss(self, x, rec):
        return self.rec_var_inv*cd(x, rec)

    def elbo_for_y(self, x, y, feats, M=1):
        _, z_mean, z_log_sigma2 = self.encode_z(y, feats)
        target_mean = self.means[torch.nonzero(y)[0,1]]
        # components_mean, components_log_sigma2 = self.get_components(y)

        KL_loss = 0.5*torch.sum(
            - 1.0
            + torch.exp(z_log_sigma2)
            - z_log_sigma2
            + ((target_mean-z_mean) ** 2)
        , dim=1)

        y_penalty = torch.full_like(KL_loss, -torch.log(torch.tensor(1/self.clusters)), device=KL_loss.device)

        rec_loss = torch.zeros_like(KL_loss, device=KL_loss.device)
        for i in range(M):
            z = self.sample(z_mean, z_log_sigma2)
            rec = self.decoder(z).view(x.shape)
            rec_loss += self.rec_loss(x, rec)
        rec_loss /= M

        return rec_loss + KL_loss + y_penalty

    def elbo_loss(self, x, M=1):
        feats = self.pointnet(x)
        log_prob_y = self.cat_encoder(feats)
        prob_y = torch.exp(log_prob_y)

        loss_list = []
        N = x.shape[0]
        for i in range(self.clusters):
            y = one_hot(torch.full((N,), i, dtype=torch.long), self.clusters).to(x.device)
            loss = self.elbo_for_y(x, y, feats, M=M)
            loss_list.append(loss)
        loss_per_y = torch.stack(loss_list, dim=1)
        loss = torch.mean(torch.sum(prob_y * loss_per_y, dim=1))
        entropy = torch.mean(torch.sum(prob_y * log_prob_y, dim=1))
        return loss + entropy, {'loss': loss.item(), 'entropy': entropy.item()}


# MNIST_LATENT = 64
# MNIST_HIDDEN = 512
# MNIST_INPUT_DIM = 784
# MNIST_CLASSES = 10

# class MNISTGMVAE(GMVAE):
#
#     DEFAULT_SAVED_NAME = 'mnistgmvae'
#
#     def __init__(self):
#         super(SaveableModule, self).__init__()
#         self.num_classes = MNIST_CLASSES
#         self.class_encoder = nn.Sequential(
#             prep_seq(MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, self.num_classes),
#             nn.LogSoftmax(dim=1),
#         )
#         self.latent_encoder = ClassMLP(self.num_classes,
#                                      MNIST_INPUT_DIM, MNIST_HIDDEN, MNIST_HIDDEN, 2*MNIST_LATENT)
#         self.decoder = nn.Sequential(
#             prep_seq(MNIST_LATENT, MNIST_HIDDEN, MNIST_HIDDEN, MNIST_INPUT_DIM),
#             nn.Sigmoid(),
#         )
#         self.gm_components = prep_seq(self.num_classes, 2*MNIST_LATENT)
#
#     def rec_loss(self, x, rec):
#         return torch.sum(F.binary_cross_entropy(rec, x, reduction='none'), dim=1)
