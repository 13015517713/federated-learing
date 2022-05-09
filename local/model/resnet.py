import torch.nn as nn
from torchvision.models.resnet import resnet18
class Model(nn.Module):
    def __init__(self, input_features=3, output_features=10):
        super().__init__()
        self.model = resnet18(num_classes=output_features)
        self.model.maxpool = nn.Identity()
    def forward(self, x):
        x = self.model(x)
        return x