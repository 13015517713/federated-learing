# Client includes dataloader and model
import torch
from torch.utils.data.dataloader import DataLoader
from util.model_util import get_flat_params_from, set_flat_params_to
class Client():
    def __init__(self, client_id, model, trainer, train_set, test_set, batch_size):
        self.id = client_id
        self.model = model
        self.trainer = trainer
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    # call model train and return model params to server
    def train(self):
        return self.trainer.self_train(self.id, self.train_loader)
    # call model test
    def test(self):
        return self.trainer.self_test(self.test_loader)
    def get_data_nums(self):
        return len(self.train_set)
    # get_model_params
    def get_flat_model_params(self):
        return get_flat_params_from(self.model)
    # set_model_params
    def set_flat_model_params(self, params):
        set_flat_params_to(self.model, params)