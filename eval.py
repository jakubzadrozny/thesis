import torch.cuda
from torch.utils.data import DataLoader

from datasets import ModelnetDataset, MNIST, FAVOURITE_CLASS, FAVOURITE_CLASSES
from vae import VAE, MNISTVAE
from m2 import M2, MNISTM2, ModifiedM2, MNISTModifiedM2
from gmvae import GMVAE, MNISTGMVAE

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def eval_supervised(model, loader):
    model.eval()
    N = 0
    score = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model.classify(x)
        score += torch.sum(pred == y)
        N += x.shape[0]
    return float(score.item()) / float(N)


def eval_unsupervised(model, loader):
    model.eval()
    probs, truth = [], []
    for x, y in loader:
        x = x.to(device)
        truth.append(y)
        probs.append(model.encode_y(x))
    probs = torch.cat(probs, dim=0)
    truth = torch.cat(truth, dim=0).to(device)

    cluster = torch.argmax(probs, dim=1)
    best = torch.argmax(probs, dim=0)
    N = cluster.shape[0]
    K = best.shape[0]
    labels = torch.zeros(N, device=device, dtype=torch.int64)
    for i in range(K):
        labels[cluster == i] = truth[best[i]]

    score = torch.sum(labels == truth)
    return float(score.item()) / float(N)


if __name__ == '__main__':
    test_dataset = MNIST(train=False)
    loader = DataLoader(test_dataset, batch_size=32, num_workers=2, drop_last=True)
    model = MNISTGMVAE.load_from_drive(MNISTGMVAE)
    print(eval_unsupervised(model, test_dataset))
