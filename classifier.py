import pytorch_lightning as pl
import torch
import torch.nn as nn

from lightly.models.utils import deactivate_requires_grad

class Classifier(pl.LightningModule):
    def __init__(self, backbone, input_dim, output_dim) :
        super().__init__()
        self.backbone = backbone

        deactivate_requires_grad(backbone)

        self.fc = nn.Linear(input_dim, output_dim)

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x) :
        x = self.backbone(x)
        y_hat = self.fc(x)
        return y_hat
    
    def training_step(self, batch, batch_idx) :
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


