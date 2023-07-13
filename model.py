import torch
import torchvision
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, projection_head, classifier, current_phase):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = projection_head
        self.classifier = classifier
        self.current_phase = current_phase

    def forward(self, x, current_phase):
        if current_phase == "self_supervised" :
            x = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(x)
        elif current_phase == "supervised" :
            with torch.no_grad():
                x = self.backbone(x).flatten(start_dim=1)
            z = self.classifier(x)
        return z
    
