import torch
import torchvision

from transforms import MyTransform

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