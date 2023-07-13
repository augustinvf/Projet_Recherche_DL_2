import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from dataloader import train_dataloader
from model import Model

nb_classes = 10

projection_head = SimCLRProjectionHead
classifier = nn.Linear(1, nb_classes)

model = Model(projection_head, classifier)

criterion_ss = NTXentLoss()
optimizer_ss = torch.optim.Adam(model.parameters(), 0.0003, weight_decay=1e-4)
scheduler_ss = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ss, T_max=len(train_dataloader), eta_min=0,
                                                           last_epoch=-1)

criterion_su = nn.CrossEntropyLoss()
optimizer_su = torch.optim.Adam(model.parameters(), 0.001, weight_decay=1e-5)
scheduler_su = torch.optim.lr_scheduler.StepLR(optimizer_su, step_size=10, gamma=0.1)


