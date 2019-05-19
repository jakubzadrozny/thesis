import torch.cuda
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from datasets import ModelnetDataset, MNIST, FAVOURITE_CLASS, FAVOURITE_CLASSES
from vae import VAE, MNISTVAE
from m2 import M2, MNISTM2, ModifiedM2, MNISTModifiedM2
from gmvae import GMVAE, MNISTGMVAE

INF = 1e60

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

def train_unsupervised(model, optimizer, loader, num_epochs=1000, M=1, lbd=0.0, log_every=200):
    print('Training your model!\n')
    model.train()
    model.to(device)

    global_step = 0
    best_params = None
    best_loss = INF

    try:
        for epoch in range(num_epochs):
            for inum, (x, _) in enumerate(loader):
                x.to(device)

                global_step += 1
                optimizer.zero_grad()

                loss, stats = model.elbo_loss(x, M=M, lbd=lbd)

                loss.backward()
                optimizer.step()

                if (global_step%log_every) == 1:
                    print("global step: %d (epoch: %d, step: %d), loss: %f %s" %
                          (global_step, epoch, inum, loss.item(), stats))

    except KeyboardInterrupt:
        pass

    model.to('cpu')
    model.eval()
    print('Saving model to drive...', end='')
    model.save_to_drive()
    print('done.')


def train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader, num_epochs=1000, p=0.0, M=1, lbd=0.0, log_every=200):
    print('Training your model!\n')
    model.train()
    model.to(device)

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
                    x.to(device)
                    y.to(device)
                except StopIteration:
                    break

                if drop_labels:
                    loss, stats = model.elbo_unknown_y(x, M=M, lbd=lbd)
                else:
                    loss, stats = model.elbo_known_y(x, y, M=M, lbd=lbd)
                loss.backward()
                optimizer.step()

                if (global_step%log_every) == 1:
                    print("global step: %d (epoch: %d), loss: %f %s" %
                          (global_step, epoch, loss.item(), stats))

    except KeyboardInterrupt:
        pass

    model.to('cpu')
    model.eval()
    print('Saving model to drive...', end='')
    model.save_to_drive()
    print('done.')


def train_vae(model, train_dataset, log_every=200):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    optimizer = Adam(model.parameters(), lr=1e-4)
    train_unsupervised(model, optimizer, train_loader, log_every=log_every)


def train_m2(model, train_dataset, drop_labels=0.0, log_every=200):
    N = len(train_dataset)
    t = int(torch.clamp(torch.tensor(N * drop_labels), 1, N-1).item())
    unlabeled_dataset, labeled_dataset = random_split(train_dataset, (t, N-t))

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

    optimizer = Adam(model.parameters(), lr=1e-4)
    train_semisupervised(model, optimizer, labeled_loader, unlabeled_loader,
                        p=drop_labels, lbd=0.0, num_epochs=2000, log_every=log_every)


if __name__ == '__main__':
    # train_dataset = ModelnetDataset(filter=FAVOURITE_CLASSES)
    # model = M2(K=len(FAVOURITE_CLASSES))
    train_dataset = MNIST()
    # model = MNISTM2()
    # train_m2(model, train_dataset, drop_labels=0.5, log_every=20)
    model = MNISTGMVAE()
    train_vae(model, train_dataset)
