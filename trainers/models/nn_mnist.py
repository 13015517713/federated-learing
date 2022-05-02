# Model is implemented for the paper, "Communication-Efficient
#                  Learning of Deep Networks from Decentralized Data."
# Code refers https://github.com/AshwinRJ/Federated-Learning-PyTorch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from util.record_util import AverageMeter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard_log')

class Model(nn.Module):
    def __init__(self, input_features=1, output_features=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_features)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Trainer():
    def __init__(self, model, epochs, lr):
        self.model = model
        self.epochs = epochs
        self.lr = lr
    def self_train(self, client_id, train_loader):
        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()
        time_recorder = AverageMeter()
        model = self.model.cuda()
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=5e-4)
        loss_func = nn.CrossEntropyLoss()
        loader = train_loader
        model.train()
        end_time = time.time()
        for epoch in range(self.epochs):
            # tensorboard need accuracy and loss
            t_loader = tqdm(enumerate(train_loader))
            for idx, (input,target) in t_loader:
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = loss_func(output, target)
                # measure accuracy and record loss per batch
                batch_size = target.size(0)
                pred = output.detach().cpu().argmax(dim=1)
                acc_recorder.update(100*(pred==target.cpu()).float().sum() / batch_size, batch_size)
                loss_recorder.update(loss.item())
                time_recorder.update(time.time()-end_time)
                end_time = time.time()
                # tqdm metric logs
                metric_str = " ".join([
                    'Train',
                    'Epoch: %d' % (epoch),
                    'Iter: [%d/%d]' % (idx + 1, len(loader)),
                    'Time %.2f (%.2f)' % (time_recorder.val, time_recorder.avg),
                    'Loss: %.4f (%.4f)'% (loss_recorder.val, loss_recorder.avg),
                    'Acc: %.4f (%.4f)' % (acc_recorder.val, acc_recorder.avg)
                ])
                t_loader.set_description(metric_str)
                # bp
                optim.zero_grad()
                loss.backward()
                optim.step()
            # tensorboard log
            '''
            writer.add_scalars(f'client_{client_id}_loss',{
                        'train_loss' : loss_recorder.avg,
                        'train_acc' :  acc_recorder.avg, 
                    }, epoch)
            '''
        self.lr = self.lr*0.992 # 权重衰减
        return loss_recorder, acc_recorder, time_recorder
    def self_test(self, test_loader):
        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()
        time_recorder = AverageMeter()
        loss_func = nn.CrossEntropyLoss()
        t_loader = tqdm(enumerate(test_loader))
        model = self.model.cuda()
        model.eval()
        end_time = time.time()
        with torch.no_grad():
            for idx, (input,target) in t_loader:
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = loss_func(output, target) # 出现问题
                pred = output.detach().cpu().argmax(dim=1) # 预测的全是一类
                batch_size = target.size(0)
                acc_recorder.update(100*(pred==target.cpu()).float().sum() / batch_size, batch_size)
                loss_recorder.update(loss.item())
                time_recorder.update(time.time()-end_time)
                end_time = time.time()
                # tqdm metric logs
                metric_str = " ".join([
                    'Test',
                    'Iter: [%d/%d]' % (idx + 1, len(test_loader)),
                    'Time %.2f (%.2f)' % (time_recorder.val, time_recorder.avg),
                    'Loss: %.4f (%.4f)'% (loss_recorder.val, loss_recorder.avg),
                    'Acc: %.4f (%.4f)' % (acc_recorder.val, acc_recorder.avg)
                ])
                t_loader.set_description(metric_str) # why update next line.
        return loss_recorder, acc_recorder, time_recorder

