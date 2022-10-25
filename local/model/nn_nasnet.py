import torch
import torch.nn as nn
from local.model.op_nasnet import *

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, steps=4):
        super(Cell, self).__init__()
        self.steps = steps
        self.C = C

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction: # 这个？
            self.geno = genotype.reduce
            self.concat = genotype.reduce_concat  # 我还是不理解这个想干嘛
        else:
            self.geno = genotype.normal
            self.concat = genotype.normal_concat
        self.compiler(C,reduction)

        self.multiplier = len(self.concat)

    def compiler(self, C, reduction): # 输入的channel和是否reduction
        self.nodes = []
        self.ops = nn.ModuleList()
        self.combs = nn.ModuleList()
        for (n1, n2, op1_name, op2_name, comb_name) in self.geno:
            self.nodes.append(n1)
            self.nodes.append(n2)

            stride1 = 2 if reduction and n1 < 2 else 1
            op1 = OPS[op1_name](C, stride1, False)
            if 'pool' in op1_name:
                op1 = nn.Sequential(op1, nn.BatchNorm2d(C, affine=False))

            stride2 = 2 if reduction and n2 < 2 else 1
            op2 = OPS[op2_name](C, stride2, False)
            if 'pool' in op2_name:
                op2 = nn.Sequential(op2, nn.BatchNorm2d(C, affine=False))

            self.ops.append(op1)
            self.ops.append(op2)

            if comb_name == 'add':
                self.combs.append(None)
            else:
                self.combs.append(ReLUConvBN(self.C * 2, self.C, 1, 1, 0))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self.steps):
            h1 = states[self.nodes[2 * i]]
            h2 = states[self.nodes[2 * i + 1]]
            op1 = self.ops[2 * i]
            op2 = self.ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            comb = self.combs[i]
            if comb == None:
                s = h1 + h2
            else:
                s = torch.cat([h1,h2], dim=1)
                s = comb(s)
            states += [s]

        return torch.cat([states[i] for i in self.concat], dim=1) # concat没有被选中的几个

class Model(nn.Module):
    # C是初始channel，这个还是比较大的，darts先取16，后面搜到后加到了36
    # 开始8个cell，后面增加到了20个
    def __init__(self, genotype, num_classes=10, C=16, stem_multiplier=3, layers=8):
        super(Model, self).__init__()  
        # genotype 这个怎么转换成模型，可以看看。
        C_curr = stem_multiplier * C    # 刚开始把channel增大
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell] # 没有被选到的都concat在一起
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

