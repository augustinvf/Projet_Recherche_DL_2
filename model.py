import torch
import torchvision
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, projection_head, classifier):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = projection_head
        self.classifier = classifier
        self.current_phase = "self_supervised"

    def forward(self, x):
        if self.current_phase == "self_supervised" :
            x = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(x)
        else :
            with torch.no_grad():
                x = self.backbone(x).flatten(start_dim=1)
                z = self.classifier(x)
        self.current_phase = "supervised" if self.current_phase == "self_supervised" else "self_supervised"
        return z
    
class SimCLRTransformation(nn.Module):
    def __init__(self, transformation1, transformation2) :
        self.transformation1 = transformation1
        self.transformation2 = transformation2
    def __call__(self, x) :
        return [self.transformation1(x), self.transformation2(x)]

