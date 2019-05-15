import torch.cuda
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from datasets import ModelnetDataset, MNIST, FAVOURITE_CLASS, FAVOURITE_CLASSES
from models import VAE, MNISTVAE, M2, MNISTM2, ModifiedM2, ENCODER_HIDDEN, DECODER_LAYERS

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
            for inum, (x, _) in enumerate(loader):
                if USE_CUDA:
                    x = x.cuda()

                global_step += 1
                optimizer.zero_grad()

                loss, stats = model.elbo_loss(x, mc_samples=mc_samples, lbd=lbd)

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

                drop_labels = torch.bernoulli(torch.tensor(p)).item()
                draw_from = unlabeled_iter if drop_labels else labeled_iter

                try:
                    x, y = next(draw_from)
                except StopIteration:
                    break
                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()

                if drop_labels:
                    loss, stats = model.elbo_unknown_y(x, mc_samples=mc_samples, lbd=lbd)
                else:
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


def train_vae(model, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    optimizer = Adam(model.parameters(), lr=1e-4)
    train_unsupervised(model, optimizer, train_loader, lbd=0.0, num_epochs=2000)


def train_m2(model, train_dataset, drop_labels=0.0):
    N = len(train_dataset)
    t = int(torch.clamp(torch.tensor(N * drop_labels), 1, N-1).item())
    unlabeled_dataset, labeled_dataset = random_split(train_dataset, (t, N-t))

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

    optimizer = Adam(model.parameters(), lr=1e-4)
    train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader,
                        p=drop_labels, lbd=0.0, num_epochs=2000)


if __name__ == '__main__':
    # train_dataset = ModelnetDataset(filter=FAVOURITE_CLASSES)
    # model = M2(train_dataset.num_classes, ENCODER_HIDDEN, DECODER_LAYERS)
    train_dataset = MNIST()
    model = MNISTM2()
    train_m2(model, train_dataset, drop_labels=0.5)
