from matplotlib.pyplot import axis
from servers.optimizers.base_fedopt import BaseFedOpt
from util.model_util import get_flat_params_from
import torch
class Optimizer(BaseFedOpt):
    def __init__(self):
        super().__init__()
    def aggerate(self, base_model, collect_params):
        sum_samples = sum([i['num_samples'] for i in collect_params])
        avg_params = torch.zeros_like(get_flat_params_from(base_model))
        for client_params in collect_params:
            avg_params += client_params['num_samples']*client_params['params']
        avg_params /= sum_samples
        return avg_params
    def aggerate(self, base_model, collect_params):
        client_samples = [i[0] for i in collect_params]
        client_params = [i[1] for i in collect_params]
        sum_samples = sum(client_samples)
        stat_dict = base_model.state_dict()
        for p_name in stat_dict:
            params = [param[p_name] for param in client_params]
            new_params = [client_samples[i]*params[i]/sum_samples 
                                    for i in range(len(collect_params))]
            stat_dict[p_name] = torch.sum(torch.stack(new_params), axis=0)
        return stat_dict