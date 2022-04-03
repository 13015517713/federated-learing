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