import numpy as np
import h5py
import gdown
import os.path
from os import listdir, mkdir

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATA_DIR = 'data'
DATA_FILE_BASE = 'ply_data_train'
DATA_FILE_EXT = '.h5'

FAVOURITE_CLASS = 8
FAVOURITE_CLASSES = [0, 8, 30]

def one_hot(y, K):
    x = torch.zeros(K)
    x[y] = 1
    return x

class FromNpDataset(Dataset):
    def __init__(self, np_data, labels, transform=None):
        self.data = np_data
        self.labels = labels
        self.num_classes = len(set(labels))
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = one_hot(self.labels[idx], self.num_classes)
        if self.transform:
            sample = self.transform(sample)
        res = torch.from_numpy(sample)
        return res, label


class ModelnetDataset(FromNpDataset):

    DATA_URLS = [
        'https://drive.google.com/uc?export=download&id=1MNgxTzGCw5By8a9aNLi7pYjcuEoRUxsR',
        'https://drive.google.com/uc?export=download&id=1co-dX33hgpDk7vUDYS-n-oOWJHSuLBkq',
        'https://drive.google.com/uc?export=download&id=1VDqD4PdqGdbfsOKdQAc4zZTsH4a1Nyqo',
        'https://drive.google.com/uc?export=download&id=1N5DlhvDQ1IkdMlpIDaKtZXqGu12ULG4F',
        'https://drive.google.com/uc?export=download&id=1UlcrapAbSBRDhCNVsuPMEaEAcvDXxOLY',
    ]

    def __init__(self, filter=None, transform=None):
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)

        for idx, url in enumerate(ModelnetDataset.DATA_URLS):
            file_path = os.path.join(DATA_DIR, DATA_FILE_BASE+str(idx)+DATA_FILE_EXT)
            if not os.path.exists(file_path):
                gdown.download(url, file_path, quiet=False)

        data_list, label_list = [], []
        h5_files = [ f for f in listdir(DATA_DIR)
                        if os.path.isfile(os.path.join(DATA_DIR, f))
                        if os.path.splitext(f)[1] == DATA_FILE_EXT ]
        for f in h5_files:
            hf = h5py.File(os.path.join(DATA_DIR, f), 'r')
            data_list.append(hf.get('data'))
            label_list.append(hf.get('label'))

        data = np.transpose(np.concatenate(data_list, axis=0), (0, 2, 1))
        labels = np.concatenate(label_list, axis=0)

        if filter:
            idx = [ i for i in range(data.shape[0]) if labels[i] in filter ]
            data = data[idx]
            labels = [ filter.index(labels[i]) for i in idx ]

        super().__init__(data, labels, transform=transform)


def MNIST(Dataset):
    def __init__(self):
        self.num_classes = 10
        self.dataset = datasets.MNIST('data/', download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x.flatten(), one_hot(y, self.num_classes)
