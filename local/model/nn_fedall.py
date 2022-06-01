import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.fc_1 = nn.Linear(10*len(self.model_list), 15, bias=True)
        self.fc_2 = nn.Linear(15, 10, bias=True)
        self.fc_3 = nn.Linear(10, 10, bias=True)
    def forward(self, x):
        with torch.no_grad():
            x_output = [model(x) for model in self.model_list]
        x = torch.cat(x_output, dim=1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x