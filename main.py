import wandb
wandb.login()

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from dataloader import train_dataloader_self_supervised, train_dataloader_supervised, test_dataloader, batch_size, train_dataset_self_supervised
from model import Model

wandb.init(
    project = "deep_learning_project_2",
    name = "Run_test_4"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(len(train_dataset_self_supervised))
print(len(train_dataloader_supervised))

nb_classes = 10
input_size_classifier = 512
projection_head = SimCLRProjectionHead(512, 512, 128)

model = Model(projection_head, input_size_classifier, nb_classes).to(device)

criterion_ss = NTXentLoss()
optimizer_ss = torch.optim.Adam(model.parameters(), 0.0003, weight_decay=1e-4)
scheduler_ss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, T_max=len(train_dataset_self_supervised), eta_min=0,
                                                           last_epoch=-1)

criterion_su = nn.CrossEntropyLoss()
optimizer_su = torch.optim.Adam(model.parameters(), 0.001, weight_decay=1e-5)
scheduler_su = torch.optim.lr_scheduler.StepLR(optimizer_su, step_size=10, gamma=0.1)

nb_epochs = 100
accuracy = 0

# training

for epochs in range(nb_epochs) :
    for mini_batch, labels in train_dataloader_self_supervised :

        # self-supervised phase

        # .to(device)
        augmented_image1 = mini_batch[0].to(device)
        augmented_image2 = mini_batch[1].to(device)

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

    for mini_batch, labels in train_dataloader_supervised :

        # supervised phase
        image_without_augmentation = mini_batch.to(device)
        labels = labels.to(device)

        y_hat = model(image_without_augmentation, "supervised")

        accuracy += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))

        loss_su = criterion_su(y_hat, labels)

        # reinitialization of the gradients
        optimizer_su.zero_grad()

        # backward propagation
        loss_su.backward()
        optimizer_su.step()
        scheduler_su.step()

    wandb.log({"loss self-supervised": loss_ss, 
               "loss supervised": loss_su,
               "accuracy": accuracy/batch_size,
            })

# test

model.eval()

total_tests = 0
positive_tests = 0

for mini_batch, labels in test_dataloader :

    mini_batch = mini_batch.to(device)
    labels = labels.to(device)

    with torch.no_grad() :
        y_hat = model(image_without_augmentation, "supervised")

    total_tests += labels.shape[0]
    positive_tests += torch.sum(torch.eq(torch.argmax(y_hat, axis = 1), labels))

wandb.log({"final accuracy" : positive_tests / total_tests})