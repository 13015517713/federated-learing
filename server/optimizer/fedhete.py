# 处理异构问题，通过不同的cell去处理

import copy
from pyparsing import alphas
import torch
import logging
import math
import random
import numpy as np
from collections import namedtuple
from local.model import op_nasnet as op
from local.model import nn_nasnet as nasnet
from server.optimizer.fedbase import BaseServer
from util.model_util import set_dict_params_to
from local.trainer import trainer_common, trainer_distill

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
cell_nums_level = [8, 16, 24, 32, 40]

def index_to_geno(actions_index):
    steps = 4
    normal = []
    reduce = []
    normal_concat = set(range(2,6))
    reduce_concat = set(range(2,6))

    for i in range(2*steps):
        node1 = int(actions_index[i*5])
        node2 = int(actions_index[i*5+1])

        op1 = op.OP_NAME[actions_index[i*5+2]]
        op2 = op.OP_NAME[actions_index[i*5+3]]

        comb = op.COMB_NAME[actions_index[i*5+4]]

        block = (node1, node2, op1, op2, comb)
        if i < steps:
            if node1 in normal_concat:
                normal_concat.remove(node1)
            if node2 in normal_concat:
                normal_concat.remove(node2)
            normal.append(block)
        else:
            if node1 in reduce_concat:
                reduce_concat.remove(node1)
            if node2 in reduce_concat:
                reduce_concat.remove(node2)
            reduce.append(block)

    genotype = Genotype(normal = normal, normal_concat = normal_concat,
                        reduce = reduce, reduce_concat = reduce_concat)

    return genotype

def getsamples(nums_sample, random_state):
    random.seed(random_state)
    
    steps = 4
    len_nodes = steps + 1
    len_OPS = len(op.OP_NAME)
    len_combs = len(op.COMB_NAME)
    nodes = list(range(len_nodes))
    OPS = list(range(len_OPS))
    combs = list(range(len_combs))

    all_genotypes = []    
    for _ in range(nums_sample):
        actions_index = []
        for type in range(2):
            for node in range(steps):
                actions_index.append(random.choice(nodes[:node+2]))
                actions_index.append(random.choice(nodes[:node+2]))
                actions_index.append(random.choice(OPS))
                actions_index.append(random.choice(OPS))
                actions_index.append(random.choice(combs))
        all_genotypes.append(index_to_geno(actions_index))

    return all_genotypes

# 客户端应该是具有不同属性的
class Server(BaseServer):
    def __init__(self, global_model, global_trainer, global_testset, clients, options):
        super().__init__(global_model, global_trainer, global_testset, clients, options)
        self.options = options
        self.client_tiers = options['client_tiers']
        cover_scale = options['cover_scale']

        # 客户端在不同簇中
        # 1.初始化不同簇的种群
        self.models = [[] for _ in range(self.client_tiers)]
        self.clients = [[] for _ in range(self.client_tiers)]
        for i in range(self.client_tiers):
            client_cnt = math.ceil(cover_scale/self.client_tiers*len(clients))
            if i == self.client_tiers-1:
                self.clients[i] = clients[i*client_cnt:]
            else:
                self.clients[i] = clients[i*client_cnt:(i+1)*client_cnt]
        
        model_num = options['model_tiers']
        for i in range(self.client_tiers):
            genotypes = getsamples(model_num, random_state=i)
            # *********************************************************************************
            # 初始化种群：存在问题，cell较小模型flop可能大于cell较大，是否考虑对tier模型设范围（条件初始化）
            # *********************************************************************************
            self.models[i] = [nasnet.Model(geno, layers=cell_nums_level[i]) for geno in genotypes]
            
        # 不同的客户端使用trainer
        common_trainer = trainer_common.Trainer(None, options) # trainer其实没必要保留model，这一点最后可以优化
        distill_trainer = trainer_distill.Trainer(None, options)
        for i in range(self.client_tiers):
            for client in self.clients[i]:
                client.trainer = common_trainer
                client.aux_trainer = distill_trainer
    def run(self):
        client_frac = self.options['client_frac']
        for round in range(self.options['rounds']):
            # 2.每个簇中的客户端进行训练
            best_model = []
            for i in range(self.client_tiers):
                # 3.双采样训练模型
                for _ in range(1):  # 默认双采样四次
                    # 采样模型
                    model_frac = 0.01
                    sel_models = random.sample(self.models[i], k=math.ceil(len(self.models[i])*model_frac) )
                    # 采样客户端：tier较小的模型可以部署到较大的客户端
                    client_list = [client for sublist in self.clients[i:] for client in sublist]
                    sel_clients = random.sample(client_list, k=math.ceil(len(client_list)*client_frac) )
                    for model in sel_models:
                        collect_params = []
                        self.global_model = model # 聚合需要模型结构
                        for client in sel_clients:
                            # 模拟发送给客户端模型
                            client.trainer.model = model
                            client.aux_trainer.model = model
                            client.model = model
                            if i == 0:
                                client.train()
                            else:
                                # 蒸馏参数设置，调优
                                client.train_distill(best_model[-1:], alpha=0.5, temperature=1)
                            num_samples, params = client.get_data_nums(), client.get_model_params()
                            collect_params.append((num_samples, params))
                        model_params = self.aggerate(collect_params)
                        set_dict_params_to(model, model_params)
                # 4.验证，排序寻找最优模型
                models_acc = []
                for model in self.models[i]:
                    acclist = []
                    for client in self.clients[i]:
                        client.trainer.model = model
                        client.model = model
                        _, acc_recorder, _ = client.test()
                        acclist.append(acc_recorder.avg)
                    acc = np.mean(acclist)
                    models_acc.append(acc)
                self.models[i] = [model for _, model in sorted(zip(models_acc, self.models[i]), reverse=True)]
                best_model.append(self.models[i][0])
            # 5.种群进化tournament
            