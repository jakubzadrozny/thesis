import numpy as np
import h5py
import gdown
import os.path
from os import listdir, mkdir

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATA_DIR = 'data'
DATA_FILE_EXT = '.h5'

PC_OUT_DIM = 3*2048

def one_hot(y, K):
    x = torch.zeros(K)
    x[y] = 1
    return x

def index_or_len(lst, item):
    try:
        return lst.index(item)
    except ValueError:
        return len(lst)


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


class PCDataset(FromNpDataset):
    def __init__(self, test=False, filter=100, transform=None):
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)

        if test:
            file_base = self.TEST_FILE_BASE
            data_urls = self.TEST_DATA_URLS
        else:
            file_base = self.TRAIN_FILE_BASE
            data_urls = self.TRAIN_DATA_URLS

        data_list, label_list = [], []
        for idx, url in enumerate(data_urls):
            file_path = os.path.join(DATA_DIR, file_base+str(idx)+DATA_FILE_EXT)

            if not os.path.exists(file_path):
                gdown.download(url, file_path, quiet=False)

            hf = h5py.File(file_path, 'r')
            data_list.append(hf.get('data'))
            label_list.append(hf.get('label'))

        data = np.transpose(np.concatenate(data_list, axis=0), (0, 2, 1))
        labels = np.concatenate(label_list, axis=0).squeeze()

        if filter <= len(self.FAVOURITE_CLASSES):
            idx = [ i for i in range(data.shape[0]) if index_or_len(self.FAVOURITE_CLASSES, labels[i]) < filter ]
            data = data[idx]
            labels = [ self.FAVOURITE_CLASSES.index(labels[i]) for i in idx ]

        super().__init__(data, labels, transform=transform)


class ModelnetDataset(PCDataset):
    FAVOURITE_CLASSES = [8, 0, 19]

    TRAIN_FILE_BASE = 'modelnet_data_train'
    TEST_FILE_BASE = 'modelnet_data_test'

    TRAIN_DATA_URLS = [
        'https://drive.google.com/uc?export=download&id=1MNgxTzGCw5By8a9aNLi7pYjcuEoRUxsR',
        'https://drive.google.com/uc?export=download&id=1co-dX33hgpDk7vUDYS-n-oOWJHSuLBkq',
        'https://drive.google.com/uc?export=download&id=1VDqD4PdqGdbfsOKdQAc4zZTsH4a1Nyqo',
        'https://drive.google.com/uc?export=download&id=1N5DlhvDQ1IkdMlpIDaKtZXqGu12ULG4F',
        'https://drive.google.com/uc?export=download&id=1UlcrapAbSBRDhCNVsuPMEaEAcvDXxOLY',
    ]

    TEST_DATA_URLS = [
        'https://drive.google.com/uc?export=download&id=1bBtvzwEfgzczoorJucXNfzGH2dXSBPZv',
        'https://drive.google.com/uc?export=download&id=1zBO0li-qwu95GleFpEzmkVcJB1V_zgks',
    ]


class ShapenetDataset(PCDataset):
    FAVOURITE_CLASSES = [4, 0, 8]

    TRAIN_FILE_BASE = 'shapenet_data_train'
    TEST_FILE_BASE = 'shapenet_data_train'

    TRAIN_DATA_URLS = [
        'https://drive.google.com/uc?export=download&id=1DSlRG-hi4pHFDEZ2naZZaxWGhEP5zaef',
        'https://drive.google.com/uc?export=download&id=11BCZDlDqYWii1oGOx3CjF45F4m0zrPj1',
        'https://drive.google.com/uc?export=download&id=1aRq7oDTA3nVcojTx7ZR5KJLcE8_nxw-f',
        'https://drive.google.com/uc?export=download&id=1PMGe0miwnkXv3ZycxoQa1k-6qhKFX5TE',
        'https://drive.google.com/uc?export=download&id=1pz1Md8Z-bPMQ1afz264z4d8DCZoXqLbq',
        'https://drive.google.com/uc?export=download&id=1KnrrvRuaizZa2GZhG0IyVPO74IIodCnt',
        'https://drive.google.com/uc?export=download&id=1cXcitlmm6IGtNdEuoTKfSg1Uzlyn-bB9',
    ]

    TEST_DATA_URLS = [
        'https://drive.google.com/uc?export=download&id=1IFl1B3F1r8tQj9JG1-50Wa02jnPOM5Rh',
        'https://drive.google.com/uc?export=download&id=11aCJmK7JyGkAsennArta1hICyaBgq0KI',
    ]


class JointDataset(FromNpDataset):
    def __init__(self, test=False, filter=100, transform_modelnet=None, transform_shapenet=None):
        self.modelnet = ModelnetDataset(test=test, filter=filter, transform=transform_modelnet)
        self.shapenet = ShapenetDataset(test=test, filter=filter, transform=transform_shapenet)

    def __len__(self):
        return len(self.modelnet) + len(self.shapenet)

    def __getitem__(self, idx):
        if idx < len(self.modelnet):
            return self.modelnet[idx]
        else:
            return self.shapenet[idx-len(self.modelnet)]


class MNIST(Dataset):
    def __init__(self, train=True):
        self.num_classes = 10
        self.dataset = datasets.MNIST('data/', download=True, train=train, transform=transforms.ToTensor())
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.train:
            y = one_hot(y, self.num_classes)
        return x.flatten(), y
