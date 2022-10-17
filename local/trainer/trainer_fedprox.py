import copy
import time
import torch
import torch.nn as nn
from util.record_util import AverageMeter

class Trainer():
    def __init__(self, model, options):
        self.model = model
        self.epochs = options['epochs']
        self.lr = options['lr']
        self.lr_decay = options['lr_decay']
        self.mu = options['fedprox_mu']
    def self_train(self, train_loader):
        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()
        time_recorder = AverageMeter()
        
        model = self.model.cuda()
        base_params = copy.deepcopy(list(model.parameters()))
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.5)
        loss_func = nn.CrossEntropyLoss()
        model.train()
        end_time = time.time()
        
        for epoch in range(self.epochs):
            # tensorboard need accuracy and loss
            for idx, (input,target) in enumerate(train_loader):
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = loss_func(output, target)
                cur_params = list()
                for i, cur_params in enumerate(list(model.parameters())):
                    loss += (self.mu/2)*torch.pow(torch.norm(base_params[i]-cur_params), 2)
                
                # measure accuracy and record loss per batch
                batch_size = target.size(0)
                pred = output.detach().cpu().argmax(dim=1)
                acc_recorder.update(100*(pred==target.cpu()).float().sum() / batch_size, batch_size)
                loss_recorder.update(loss.item())
                time_recorder.update(time.time()-end_time)
                end_time = time.time()
                '''# tqdm metric logs
                metric_str = " ".join([
                    'Train',
                    'Epoch: %d' % (epoch),
                    'Iter: [%d/%d]' % (idx + 1, len(loader)),
                    'Time %.2f (%.2f)' % (time_recorder.val, time_recorder.avg),
                    'Loss: %.4f (%.4f)'% (loss_recorder.val, loss_recorder.avg),
                    'Acc: %.4f (%.4f)' % (acc_recorder.val, acc_recorder.avg)
                ])
                # t_loader.set_description(metric_str)'''

                # bp
                optim.zero_grad()
                loss.backward()
                optim.step()
        self.lr = self.lr*self.lr_decay
        return loss_recorder, acc_recorder, time_recorder

    def self_test(self, test_loader):
        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()
        time_recorder = AverageMeter()
        loss_func = nn.CrossEntropyLoss()
        model = self.model.cuda()
        model.eval()
        end_time = time.time()
        with torch.no_grad():
            for idx, (input,target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = loss_func(output, target)
                
                pred = output.detach().cpu().argmax(dim=1)
                batch_size = target.size(0)
                acc_recorder.update(100*(pred==target.cpu()).float().sum() / batch_size, batch_size)
                loss_recorder.update(loss.item())
                time_recorder.update(time.time()-end_time)
                end_time = time.time()
                '''# tqdm metric logs
                metric_str = " ".join([
                    'Test',
                    'Iter: [%d/%d]' % (idx + 1, len(test_loader)),
                    'Time %.2f (%.2f)' % (time_recorder.val, time_recorder.avg),
                    'Loss: %.4f (%.4f)'% (loss_recorder.val, loss_recorder.avg),
                    'Acc: %.4f (%.4f)' % (acc_recorder.val, acc_recorder.avg)
                ])'''
        return loss_recorder, acc_recorder, time_recorder
