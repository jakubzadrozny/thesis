import torch.cuda
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import FromNpDataset, ModelnetDataset, FAVOURITE_CLASS, FAVOURITE_CLASSES
from models import VAE, M2, ENCODER_HIDDEN, DECODER_LAYERS

INF = 1e60
USE_CUDA = torch.cuda.is_available()

def train_unsupervised(model, optimizer, loader, mc_samples=1, lbd=0.0, num_epochs=100):
    print('Training your model!\n')
    model.train()
    if USE_CUDA:
        model.cuda()

    global_step = 0
    best_params = None
    best_loss = INF

    try:
        for epoch in range(num_epochs):
            for inum, batch in enumerate(loader):
                if USE_CUDA:
                    batch = batch.cuda()

                global_step += 1
                optimizer.zero_grad()

                loss, stats = model.elbo_loss(batch, mc_samples=mc_samples, lbd=lbd)

                loss.backward()
                optimizer.step()

                if (global_step%200) == 1:
                    print("global step: %d (epoch: %d, step: %d), loss: %f %s" %
                          (global_step, epoch, inum, loss.item(), stats))

    except KeyboardInterrupt:
        pass

    model.to('cpu')
    model.eval()
    print('Saving model to drive...', end='')
    model.save_to_drive()
    print('done.')


def train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader, p=0.0, mc_samples=1, lbd=0.0, num_epochs=100):
    print('Training your model!\n')
    model.train()
    if USE_CUDA:
        model.cuda()

    global_step = 0
    best_params = None
    best_loss = INF

    try:
        for epoch in range(num_epochs):
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            while True:
                global_step += 1
                optimizer.zero_grad()
                if torch.bernoulli(torch.tensor(p)).item():
                    try:
                        x = next(unlabeled_iter)
                    except StopIteration:
                        break
                    if USE_CUDA:
                        x = x.cuda()
                    loss, stats = model.elbo_unknown_y(x, mc_samples=mc_samples, lbd=lbd)
                else:
                    try:
                        x, y = next(labeled_iter)
                    except StopIteration:
                        break
                    if USE_CUDA:
                        x = x.cuda()
                        y = y.cuda()
                    loss, stats = model.elbo_known_y(x, y, mc_samples=mc_samples, lbd=lbd)
                loss.backward()
                optimizer.step()

                if (global_step%200) == 1:
                    print("global step: %d (epoch: %d), loss: %f %s" %
                          (global_step, epoch, loss.item(), stats))

    except KeyboardInterrupt:
        pass

    model.to('cpu')
    model.eval()
    print('Saving model to drive...', end='')
    model.save_to_drive()
    print('done.')


def train_vae():
    train_dataset = ModelnetDataset(filter=[FAVOURITE_CLASS])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    model = VAE(ENCODER_HIDDEN, DECODER_LAYERS)
    optimizer = Adam(model.parameters(), lr=2e-4)

    train_unsupervised(model, optimizer, train_loader, lbd=1.0, num_epochs=2000)


def train_m2(drop_labels=0.0):
    K = len(FAVOURITE_CLASSES)

    train_dataset = ModelnetDataset(with_labels=True, filter=FAVOURITE_CLASSES)
    N = len(train_dataset)
    t = int(N * drop_labels)
    unlabeled_dataset = FromNpDataset(train_dataset.data[:min(N, t+1)])
    labeled_dataset = FromNpDataset(train_dataset.data[max(0, t-1):], labels=train_dataset.labels[max(0, t-1)::])

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

    model = M2(K, ENCODER_HIDDEN, DECODER_LAYERS)
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader, p=drop_labels, lbd=3.0, num_epochs=2000)


if __name__ == '__main__':
    train_m2(drop_labels=0.5)
