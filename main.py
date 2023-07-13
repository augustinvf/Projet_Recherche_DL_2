import wandb
wandb.login()

import torch

from dataloader import train_dataloader, test_dataloader
from utils import model, criterion_ss, optimizer_ss, scheduler_ss, criterion_su, optimizer_su, scheduler_su

wandb.init(
    project = "deep_learning_project_2",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

nb_epochs = 1
c = 0

# training

for epochs in range(nb_epochs) :
    for mini_batch, labels in train_dataloader :

        # self-supervised phase

        # .to(device)
        augmented_image1 = mini_batch[1].to(device)
        augmented_image2 = mini_batch[2].to(device)

        # forward propagation for both images
        y_hat_1 = model(augmented_image1, "self-supervised")
        y_hat_2 = model(augmented_image2, "self-supervised")

        # loss calculation
        loss_ss = criterion_ss(y_hat_1, y_hat_2)

        # reinitialization of the gradients
        optimizer_ss.zero_grad()

        # backward propagation
        loss_ss.backward()
        optimizer_ss.step()
        scheduler_ss.step()

        # supervised phase

        image_without_augmentation = mini_batch[0].to(device)
        labels = labels.to(device)

        y_hat = model(image_without_augmentation, "supervised")

        accuracy = torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels)) / y_hat.shape[0]

        loss_su = criterion_su(y_hat, labels)

        # reinitialization of the gradients
        optimizer_su.zero_grad()

        # backward propagation
        loss_su.backward()
        optimizer_su.step()
        scheduler_su.step()

        wandb.log({"loss self-supervised": loss_ss, "loss supervised": loss_su, "accuracy": accuracy})

        c += 1

        if c == 6 :

            break

# test

total_tests = 0
positive_tests = 0

for mini_batch, labels in test_dataloader :

    mini_batch = mini_batch.to(device)
    labels = labels.to(device)

    y_hat = model(image_without_augmentation, "supervised")

    total_tests += labels.shape
    positive_tests += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))



wandb.log({"final accuracy" : positive_tests / total_tests})