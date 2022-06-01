import torch
import torch.nn as nn
from local.model.nn_cifar import Model as model_cifar
from local.model.nn_mnist import Model as model_mnist

class Model(nn.Module):
    def __init__(self, model_source_list, model_type_list):
        super().__init__()
        base_model = [model_cifar() if i == 'nn_cifar'  \
                        else model_mnist() for i in model_type_list]
        for i, model_src in enumerate(model_source_list):
            base_model[i].load_state_dict(torch.load(model_src))
        self.model_list = base_model
        for model in self.model_list:
            model.cuda()
            for param in model.parameters(): param.requires_grad = False
    def forward(self, x):
        with torch.no_grad():
            x_output = [model(x) for model in self.model_list]
        x = torch.stack(x_output)
        x = torch.sum(x, axis=0)/len(x_output)
        return x