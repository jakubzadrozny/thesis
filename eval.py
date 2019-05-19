import torch.cuda
from torch.utils.data import DataLoader

from datasets import ModelnetDataset, MNIST, FAVOURITE_CLASS, FAVOURITE_CLASSES
from vae import VAE, MNISTVAE
from m2 import M2, MNISTM2, ModifiedM2, MNISTModifiedM2

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

def eval_model(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, num_workers=2, drop_last=True)
    N = 0
    score = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model.classify(x)
        score += torch.sum(pred == y)
        N += x.shape[0]
    return float(score.item()) / float(N)

if __name__ == '__main__':
    test_dataset = MNIST(train=False)
    model = MNISTM2.load_from_drive(MNISTM2)
    print(eval_model(model, test_dataset))
