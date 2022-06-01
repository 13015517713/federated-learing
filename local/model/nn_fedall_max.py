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
    def forward(self, x):
        with torch.no_grad():
            x_output = [F.softmax(model(x), dim=1) for model in self.model_list]
        # x_output : [k*128*10]
        x_temp = torch.stack(x_output, dim=0)
        # x_temp: [k*128*10]
        x_temp,_ = torch.max(x_temp, dim=2)
        # x_temp: [k*128]
        _, idx = torch.max(x_temp, dim=0)
        # idx: [128]:k
        x = torch.empty_like(x_output[0])
        # x: 128*10
        for i in range(x.shape[0]):
            x[i] = x_output[idx[i]][i]
        return x
    # 需要的是softmax之后的最大值