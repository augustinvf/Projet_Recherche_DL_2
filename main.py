import wandb
wandb.login()

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from dataloader import train_dataloader_self_supervised, train_dataloader_supervised, test_dataloader, batch_size
from model import Model

wandb.init(
    project = "deep_learning_project_2",
    name = "Run_test_4"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_classes = 10
input_size_classifier = 512
projection_head = SimCLRProjectionHead(512, 512, 128)
nb_steps = len(train_dataloader_supervised)
nb_epochs = 100

model = Model(projection_head, input_size_classifier, nb_classes).to(device)

criterion_ss = NTXentLoss()
optimizer_ss = torch.optim.Adam(model.parameters(), 0.0003, weight_decay=1e-4)
scheduler_ss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, T_max=nb_steps*nb_epochs, eta_min=0,
                                                           last_epoch=-1)

criterion_su = nn.CrossEntropyLoss()
optimizer_su = torch.optim.Adam(model.parameters(), 0.001, weight_decay=1e-5)
scheduler_su = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, T_max=nb_steps*nb_epochs, eta_min=0,
                                                           last_epoch=-1)

print(scheduler_ss.get_last_lr())
print(type(scheduler_ss.get_last_lr()))
print(type(scheduler_ss.get_last_lr()[0]))

# training

for epochs in range(nb_epochs) :
    sum_loss_ss = 0
    sum_loss_su = 0
    accuracy = 0
    for mini_batch, labels in train_dataloader_self_supervised :

        # reinitialization of the gradients
        optimizer_ss.zero_grad()

        # self-supervised phase

        # .to(device)
        augmented_image1 = mini_batch[0].to(device)
        augmented_image2 = mini_batch[1].to(device)

        # forward propagation for both images
        y_hat_1 = model(augmented_image1, "self-supervised")
        y_hat_2 = model(augmented_image2, "self-supervised")

        # loss calculation
        loss_ss = criterion_ss(y_hat_1, y_hat_2)
        sum_loss_ss += loss_ss

        # backward propagation
        loss_ss.backward()
        optimizer_ss.step()
        scheduler_ss.step()

    for mini_batch, labels in train_dataloader_supervised :

        # reinitialization of the gradients
        optimizer_su.zero_grad()

        # supervised phase
        image_without_augmentation = mini_batch.to(device)
        labels = labels.to(device)

        y_hat = model(image_without_augmentation, "supervised")

        accuracy += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))

        loss_su = criterion_su(y_hat, labels)
        sum_loss_su += loss_su

        # backward propagation
        loss_su.backward()
        optimizer_su.step()
        scheduler_su.step()

    wandb.log({"loss self-supervised": sum_loss_ss/nb_steps, 
               "loss supervised": sum_loss_su/nb_steps,
               "accuracy": accuracy/(batch_size*nb_steps),
               "lr": scheduler_ss.get_last_lr()[0]
            })

# test

model.eval()

total_tests = 0
positive_tests = 0

for mini_batch, labels in test_dataloader :

    image_without_augmentation = mini_batch.to(device)
    labels = labels.to(device)

    with torch.no_grad() :
        y_hat = model(image_without_augmentation, "supervised")

    total_tests += labels.shape[0]
    positive_tests += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))

wandb.log({"final accuracy" : positive_tests / total_tests})