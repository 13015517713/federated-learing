# Client includes dataloader and model
import torch
from torch.utils.data.dataloader import DataLoader
from util.model_util import get_dict_params_from, set_dict_params_to
class Client():
    def __init__(self, client_id, model, trainer, train_set, test_set, batch_size):
        self.id = client_id
        self.model = model
        self.trainer = trainer
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)
        self.aux_trainer = None # 辅助蒸馏
    def train(self):
        return self.trainer.self_train(self.train_loader)
    def train_distill(self, teacher_models, alpha, temperature):
        return self.aux_trainer.self_train(self.train_loader, teacher_models, alpha, temperature)
    def test(self):
        return self.trainer.self_test(self.test_loader)
    def get_data_nums(self):
        return len(self.train_loader.dataset)
    def get_model_params(self):
        return get_dict_params_from(self.model)
    def set_model_params(self, params):
        set_dict_params_to(self.model, params)