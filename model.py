import torch
import torchvision
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, projection_head, input_size_classifier, nb_classes):
        super().__init__()
        resnet = torchvision.models.resnet18()
        resnet.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = projection_head
        self.classifier = nn.Linear(input_size_classifier, nb_classes)

    def forward(self, x, current_phase):
        if current_phase == "self-supervised" :
            x = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(x)
        elif current_phase == "supervised" :
            with torch.no_grad() :
                x = self.backbone(x).flatten(start_dim=1)
            z = self.classifier(x)
        return z
    
