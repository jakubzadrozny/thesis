import torch.cuda
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import ModelnetDataset, FAVOURITE_CLASS
from models import VAE, ENCODER_HIDDEN, DECODER_LAYERS

INF = 1e60
USE_CUDA = torch.cuda.is_available()
MULTIPLE_GPUS = torch.cuda.device_count() > 1

def train_model(model, optimizer, loader, mc_samples=1, beta=1.0, lbd=0.0, num_epochs=100):
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

                loss, stats = model.elbo_loss(batch, mc_samples=mc_samples, beta=beta, lbd=lbd)

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

if __name__ == '__main__':
    train_dataset = ModelnetDataset(classes=[FAVOURITE_CLASS])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = VAE(ENCODER_HIDDEN, DECODER_LAYERS)
    if MULTIPLE_GPUS:
        model = nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=2e-4)

    train_model(model, optimizer, train_loader, lbd=10.0, mc_samples=10, num_epochs=3000)
