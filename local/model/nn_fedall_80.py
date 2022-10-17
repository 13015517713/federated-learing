import torch
import torch.nn as nn
import torch.nn.functional as F
from local.model.nn_shakespeare import Model as model_shakespeare

class Model(nn.Module):
    def __init__(self, model_source_list, model_type_list):
        super().__init__()
        base_model = [model_shakespeare() for _ in range(len(model_type_list))]
        for i, model_src in enumerate(model_source_list):
            base_model[i].load_state_dict(torch.load(model_src))
        self.model_list = base_model
        for model in self.model_list:
            model.cuda()
            for param in model.parameters(): param.requires_grad = False
        self.fc_1 = nn.Linear(80*len(self.model_list), 180, bias=True)
        self.fc_2 = nn.Linear(180, 120, bias=True)
        self.fc_3 = nn.Linear(120, 80, bias=True)
    def forward(self, x):
        with torch.no_grad():
            x_output = [model(x) for model in self.model_list]
        x = torch.cat(x_output, dim=1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x