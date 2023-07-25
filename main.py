import wandb
wandb.login()

import omegaconf

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from model import Model
from dataloader import train_dataloader_self_supervised, train_dataloader_supervised, test_dataloader, batch_size
from training import self_supervised_training, supervised_training
from eval import test_fct

# To print the results

wandb.init(
    project = "deep_learning_project_2",
    name = "Run_test_5"
)

# To load the hyperparameters

config = omegaconf.OmegaConf.load("config.yaml")

# tool initialization

print(config.scheduler_su.T_max)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_classes = config.nb_classes
input_size_classifier = config.input_size_classifier
projection_head = SimCLRProjectionHead(512, 512, 128)
nb_steps = len(train_dataloader_supervised)

nb_cycles = config.nb_cycles
nb_epochs_self_supervised_by_cycle = config.nb_epochs_self_supervised_by_cycle
nb_epochs_supervised_by_cycle = config.nb_epochs_supervised_by_cycle

model = Model(projection_head, input_size_classifier, nb_classes).to(device)

criterion_ss = NTXentLoss()
optimizer_ss = torch.optim.SGD(list(model.backbone.parameters()) + list(model.projection_head.parameters()), 
                               config.optimizer_ss.lr, config.optimizer_ss.momentum, config.optimizer_ss.weight_decay)
scheduler_ss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, config.scheduler_ss.T_max, config.scheduler_ss.eta_min,
                                                           last_epoch=-1)

criterion_su = nn.CrossEntropyLoss()
optimizer_su = torch.optim.SGD(model.classifier.parameters(), config.optimizer_su.lr, 
                               config.optimizer_su.momentum, config.optimizer_su.weight_decay)
scheduler_su = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_su, config.scheduler_su.T_max, config.scheduler_su.eta_min,
                                                           last_epoch=-1)

# training

for cycles in range (nb_cycles) :
    for epochs in range(nb_epochs_self_supervised_by_cycle) :
        sum_loss_ss = self_supervised_training(device, model, train_dataloader_self_supervised, criterion_ss, optimizer_ss, scheduler_ss)
        wandb.log({"loss self-supervised": sum_loss_ss/nb_steps,
                   "learning rate self-supervised": scheduler_ss.get_last_lr()[0]
                })
    for epochs in range(nb_epochs_supervised_by_cycle) :
        sum_loss_su, accuracy = supervised_training(device, model, train_dataloader_supervised, criterion_su, optimizer_su, scheduler_su)
        wandb.log({"loss supervised": sum_loss_su/nb_steps,
               "accuracy supervised": accuracy/(batch_size*nb_steps),
               "learning rate supervised": scheduler_su.get_last_lr()[0]
                })

# test

final_accuracy = test_fct(device, model, test_dataloader)
wandb.log({"final_accuracy": final_accuracy})
