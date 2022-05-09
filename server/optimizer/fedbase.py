import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from util.model_util import get_dict_params_from
class BaseServer():
    def __init__(self, global_model, global_trainer, global_testset, clients, options):
        self.global_model = global_model
        self.global_trainer = global_trainer
        self.clients = clients
        self.options = options
        self.test_loader = DataLoader(global_testset, batch_size=128, shuffle=True, num_workers=0)
        self.lastest_params = get_dict_params_from(global_model)
    def aggerate(self, collect_params):
        client_samples = [i[0] for i in collect_params]
        client_params = [i[1] for i in collect_params]
        sum_samples = sum(client_samples)
        stat_dict = self.global_model.state_dict()
        for p_name in stat_dict:
            params = [param[p_name] for param in client_params]
            new_params = [client_samples[i]*params[i]/sum_samples 
                                    for i in range(len(collect_params))]
            stat_dict[p_name] = torch.sum(torch.stack(new_params), axis=0)
        return stat_dict
    def select_clients(self):
        frac = self.options['client_frac']
        select_nums = max(int(frac*len(self.clients)), 1)
        select_idx = np.random.choice(range(len(self.clients)), select_nums, replace=False)
        clients = [self.clients[i] for i in select_idx]
        return clients