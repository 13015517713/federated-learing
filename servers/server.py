import logging
import copy
from util.record_util import AverageMeter
from torch.utils.data.dataloader import DataLoader
from util.model_util import get_flat_params_from, set_flat_params_to
# Server includes clients, aggerator(optim), global model and global dataset
## global trainer is just in order to test global_model
class Server():
    def __init__(self, global_model, global_trainer, global_testset, clients, optim, options):
        self.global_model = global_model
        self.global_trainer = global_trainer
        self.clients = clients
        self.optim = optim
        self.options = options
        self.test_loader = DataLoader(global_testset, batch_size=options['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        self.lastest_params = get_flat_params_from(global_model)
    def run(self):
        rounds = self.options['rounds']
        mode = self.options['mode']
        logging.info(f"server communicates with clients[nums={len(self.clients)},rounds={rounds}].")
        for i in range(rounds):
            # test global model and test client model 
            if i % self.options['eval_round_nums'] == 0:
                loss_recorder, acc_recorder, _ = self.global_trainer.self_test(self.test_loader)
                logging.info("global model test, loss=%.4f, acc=%.4f."%(loss_recorder.avg, acc_recorder.avg) )
                loss_from_clients = []
                acc_from_clients = []
                for client_id, client in enumerate(self.clients):
                    loss, acc, _ = client.test()
                    loss_from_clients.append(loss.avg)
                    acc_from_clients.append(acc.avg)
                    logging.info("client_%d model test, loss=%.4f, acc=%.4f", client_id, loss.avg, acc.avg)
                logging.info("client models test, loss_avg=%.4f, acc_avg=%.4f."
                             %(sum(loss_from_clients)/len(self.clients), sum(acc_from_clients)/len(self.clients) ))
            # client train and return params
            collect_params = []
            for client in self.clients:
                client.set_flat_model_params(self.lastest_params)
                client.train()
                num_samples, params = client.get_data_nums(), client.get_flat_model_params()
                collect_params.append({'num_samples': num_samples, 'params': params})
            # optim aggerate params and return new params
            self.lastest_params = self.optim.aggerate(self.global_model, collect_params)
            # update client model
            set_flat_params_to(self.global_model, self.lastest_params)