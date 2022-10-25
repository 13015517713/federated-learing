import copy
import numpy as np
def get_dict_params_from(model):
    if model == None:
        return None
    return copy.deepcopy(model.state_dict())

def set_dict_params_to(model, dict_params):
    model.load_state_dict(dict_params)