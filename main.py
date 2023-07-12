import torch
import torchvision
import torchvision.transforms as transforms

from SimCLR import Model, SimCLRTransformation
from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
cycle_number = 5
devices = 1
nb_classes = 10

contrastive_transformations = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root='./data_cifar10_train',
    train=True,
    transform=SimCLRTransformation(contrastive_transformations, contrastive_transformations),
    download=True
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)





