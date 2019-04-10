from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import FromNpDataset, ModelnetDataset
from transforms import RandomRotation
from models import VAE, cd, elbo_loss, ENCODER_HIDDEN, DECODER_LAYERS


INF = 1000 * 1000 * 1000

def train_model(model, optimizer, loader, num_epochs=100, use_cuda=True):
    print('Training your model!\n')
    model.train()
    if use_cuda:
        model.to('cuda')

    global_step = 0
    best_params = None
    best_loss = INF

    try:
        for epoch in range(num_epochs):
            for inum, batch in enumerate(loader):
                if use_cuda:
                    batch = batch.cuda()

                global_step += 1
                optimizer.zero_grad()

                reconstruction, mu, sigma2 = model(batch)
                loss, stats = elbo_loss(batch, reconstruction, mu, sigma2, beta=0.001)

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
    train_dataset = ModelnetDataset(transform=RandomRotation())
    test_dataset = ModelnetDataset(transform=None)

    train_loader = DataLoader(train_dataset, batch_size=24,
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=24,
                            shuffle=True, num_workers=1)

    model = VAE(ENCODER_HIDDEN, DECODER_LAYERS)
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_model(model, optimizer, train_loader, num_epochs=400, use_cuda=True)
