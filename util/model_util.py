import torch
import numpy as np
# not copy, param.data have been escaped computer graph
def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params
# data.copy_, data device hasn't been changed 
def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for name,param in model.named_parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

# if there are batchnorm layers which include dynamic params without in model.parameters(),
# you shouldn't use methods about flat params. State_dict is recommended.
def get_dict_params_from(model):
    return model.state_dict()

def set_dict_params_to(model, dict_params):
    model.load_state_dict(dict_params)