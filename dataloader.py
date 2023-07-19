import torch
import torchvision

from transforms import MyTransform, basic_transformation

batch_size = 512

train_dataset_self_supervised = torchvision.datasets.CIFAR10(
    root='./data_cifar10_train',
    train=True,
    transform=MyTransform(input_size=32, gaussian_blur=0.5),
    download=True
)

train_dataloader_self_supervised = torch.utils.data.DataLoader(
    dataset=train_dataset_self_supervised,
    batch_size=batch_size,
    shuffle=True,
    num_workers=12,
    drop_last=True,
    pin_memory=True
)

train_dataset_supervised = torchvision.datasets.CIFAR10(
    root='./data_cifar10_train',
    train=True,
    transform=basic_transformation,
    download=True
)

train_dataloader_supervised= torch.utils.data.DataLoader(
    dataset=train_dataset_supervised,
    batch_size=batch_size,
    shuffle=True,
    num_workers=12,
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
    shuffle=False,
    num_workers=12,
    drop_last=True,
    pin_memory=True
)

