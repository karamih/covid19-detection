# IN this module we create our dataset and turn those to data loader.


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def creat_dataloader(train_dir: str,
                     test_dir: str,
                     batch_size: int):
    train_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader, train_dataset.classes
