import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.record_util import AverageMeter

class Trainer():
    def __init__(self, model, options):
        self.model = model
        self.epochs = options['epochs']
        self.lr = options['lr']
        self.lr_decay = options['lr_decay']
    def self_train_distill(self, train_loader, teacher_models, alpha, T):
        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()
        time_recorder = AverageMeter()

        model = self.model.cuda()
        for t_model in teacher_models: # 目前只用到第一个教师模型
            t_model.cuda()
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.5)
        loss_func = nn.CrossEntropyLoss()
        distill_loss_func = nn.KLDivLoss()
        model.train()
        end_time = time.time()
        
        for epoch in range(self.epochs):
            
            for idx, (input,target) in enumerate(train_loader):
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = (1-alpha)*loss_func(output, target)
                
                with torch.no_grad():
                    techer_output = teacher_models[0](input)
                loss += nn.KLDivLoss()(F.log_softmax(output/T, dim=1),
                             F.softmax(techer_output/T, dim=1)) * (alpha * T * T) 
                
                # measure accuracy and record loss per batch
                batch_size = target.size(0)
                pred = output.detach().cpu().argmax(dim=1)
                acc_recorder.update(100*(pred==target.cpu()).float().sum() / batch_size, batch_size)
                loss_recorder.update(loss.item())
                time_recorder.update(time.time()-end_time)
                end_time = time.time()
                
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
                ])
                # t_loader.set_description(metric_str)'''
        return loss_recorder, acc_recorder, time_recorder

