import torch
import torch.nn as nn
from local.model.nn_cifar import Model as model_cifar
from local.model.nn_mnist import Model as model_mnist

class Model(nn.Module):
    def __init__(self, model_source, model_type):
        super().__init__()
        self.base_model = model_cifar() if model_type[0] == 'nn_cifar'  \
                        else model_mnist()
        self.base_model.load_state_dict(torch.load(model_source[0]))
        self.base_model.cuda()
        for param in self.base_model.parameters(): param.requires_grad = False
    def forward(self, x):
        return self.base_model(x)