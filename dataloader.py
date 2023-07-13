import torch
import torchvision

from transforms import MyTransform, basic_transformation

batch_size = 128

train_dataset = torchvision.datasets.CIFAR10(
    root='./data_cifar10_train',
    train=True,
    transform=MyTransform(),
    download=True
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data_cifar10_test',
    train=True,
    transform=basic_transformation,
    download=True
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

