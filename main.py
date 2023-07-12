import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms

from SimCLR import SimCLR
from classifier import Classifier

batch_size = 128
cycle_number = 1
epoch_at_each_cycle = 1
devices = 1
accelerator = "gpu"
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
    transform=contrastive_transformations,
    download=True
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

contrastive_model = SimCLR()
classifier = Classifier(contrastive_model.backbone, 1, nb_classes)

for n in range (cycle_number) :
    trainer = pl.Trainer(max_epochs = epoch_at_each_cycle, devices = 1, accelerator = accelerator)
    trainer.fit(contrastive_model, train_dataloader)
    trainer.fit(classifier, train_dataloader)

