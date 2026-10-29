import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir="data", batch_size=32, train=True):
    """Create a DataLoader for MNIST, with pixels normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # [0, 1] -> [-1, 1]
    ])
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
