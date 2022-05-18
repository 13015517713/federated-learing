# Model is implemented for the paper, "Communication-Efficient
#                  Learning of Deep Networks from Decentralized Data."
# Code refers https://github.com/AshwinRJ/Federated-Learning-PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_features=1, output_features=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() 
        self.fc1 = nn.Linear(320, 50, bias=False)
        self.fc2 = nn.Linear(50, output_features, bias=False)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def wider(self):
        # fc3后面增加单位全连接层
        cur_lin = nn.Linear(50, 50, bias=False)
        with torch.no_grad():
            cur_lin.weight.copy_(torch.eye(50))
            cur_lin.to(device=self.fc2.weight.device)
        self.fc2 = nn.Sequential(
            cur_lin,
            self.fc2
        )
    def dewider(self):
        cur_lin = nn.Linear(50, 10, bias=False)
        tensor1 = self.fc2[0].weight.detach().cpu()
        tensor2 = self.fc2[1].weight.detach().cpu()
        with torch.no_grad():
            cur_tensor = torch.mm(tensor2, tensor1)
            cur_lin.weight.copy_(cur_tensor)
            cur_lin.to(device=self.fc2[0].weight.device)
        self.fc2 = cur_lin
        