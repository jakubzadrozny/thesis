import torch.cuda
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import ModelnetDataset, FAVOURITE_CLASS
from models import VAE, ENCODER_HIDDEN, DECODER_LAYERS

INF = 1e60
USE_CUDA = torch.cuda.is_available()
# MULTIPLE_GPUS = torch.cuda.device_count() > 1

def train_model(model, optimizer, loader, with_labels=False, p=0.0, mc_samples=1, lbd=0.0, num_epochs=100):
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
                x = batch[0] if with_labels else batch
                y = batch[1] if with_labels else None

                global_step += 1
                optimizer.zero_grad()

                # if MULTIPLE_GPUS:
                    # loss, stats = model.module.elbo_loss(batch, mc_samples=mc_samples, lbd=lbd)
                # else:
                if with_labels:
                    stats = {}
                    if torch.bernoulli(torch.tensor(p)).item():
                        loss = model.elbo_unknown_y(x, mc_samples=mc_samples, lbd=lbd)
                    else:
                        loss = model.elbo_known_y(x, y, mc_samples=mc_samples, lbd=lbd)
                else:
                    loss, stats = model.elbo_loss(x, mc_samples=mc_samples, lbd=lbd)
                    # loss, stats = model.elbo_loss(batch, mc_samples=mc_samples, lbd=lbd)

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


def train_vae():
    train_dataset = ModelnetDataset(filter=[FAVOURITE_CLASS])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = VAE(ENCODER_HIDDEN, DECODER_LAYERS)
    # if MULTIPLE_GPUS:
    #     model = nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=2e-4)

    train_model(model, optimizer, train_loader, lbd=0.5, num_epochs=2000)


def train_m2():
    K = len(FAVOURITE_CLASSES)
    train_dataset = ModelnetDataset(with_labels=True, filter=FAVOURITE_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = M2(K, ENCODER_HIDDEN, DECODER_LAYERS)
    # if MULTIPLE_GPUS:
    #     model = nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=2e-4)

    train_model_with_labels(model, optimizer, train_loader, with_labels=True,
                            p=0.0, lbd=1.0, num_epochs=2000)


if __name__ == '__main__':
    train_vae()
