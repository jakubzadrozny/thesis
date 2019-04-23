from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import ModelnetDataset, FAVOURITE_CLASS
from models import VAE, ENCODER_HIDDEN, DECODER_LAYERS
import transforms

INF = 1e60

def train_model(model, optimizer, loader, mc_samples=1, beta=1.0, num_epochs=100, use_cuda=True):
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

                loss, stats = model.elbo_loss(batch, mc_samples=mc_samples, beta=beta)

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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    model = VAE(ENCODER_HIDDEN, DECODER_LAYERS)
    optimizer = Adam(model.parameters(), lr=2e-4)

    train_model(model, optimizer, train_loader, num_epochs=1000, use_cuda=True)
