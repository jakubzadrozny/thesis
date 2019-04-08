from torch.utils.data import DataLoader

from datasets import FromNpDataset, ModelnetDataset
from transforms import RandomRotation
from model import VAE, cd, elbo_loss


LATENT = 50
ENCODER_HIDDEN = 512
DECODER_HIDDEN = 256
OUT_POINTS = 2048

def train_model(model, optimizer, loader, num_epochs=100):
    model.train()
    model.to('cuda')

    global_step = 0
    best_params = None
    best_loss = np.inf

    try:
        for epoch in range(num_epochs):
            for inum, batch in enumerate(loader):
                batch = batch.cuda()

                global_step += 1
                optimizer.zero_grad()

                reconstruction, mu, sigma2 = model(batch)
                loss, stats = elbo_loss(batch, reconstruction, mu, sigma2, beta=0.0)

                loss.backward()
                optimizer.step()

                if (global_step%100) == 1:
                    if loss < best_loss:
                        best_loss = loss
                        best_params = [p.detach().cpu() for p in model.parameters()]
                    print("global step: %d (epoch: %d, step: %d), loss: %f %s" %
                          (global_step, epoch, inum, loss.item(), stats))

    except KeyboardInterrupt:
        pass

    if best_params is not None:
        print("\nLoading best params")
        model.parameters = best_params
        print("Params loaded")

    model.to('cpu')
    model.eval()
    model.save_to_drive('vae_trained')

if __name__ == '__main__':
    train_dataset = ModelnetDataset(transform=RandomRotation())
    test_dataset = ModelnetDataset(transform=None)

    train_loader = DataLoader(train_dataset, batch_size=24,
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=24,
                            shuffle=True, num_workers=1)

    model = VAE(ENCODER_HIDDEN, [LATENT, DECODER_HIDDEN, 3*OUT_POINTS])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, optimizer, train_loader)
