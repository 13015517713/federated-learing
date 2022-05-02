import logging
import copy
import numpy as np
from util.record_util import AverageMeter
from torch.utils.data.dataloader import DataLoader
from util.model_util import get_flat_params_from, set_flat_params_to
from util.model_util import get_dict_params_from, set_dict_params_to
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('one_class_tensorboard_log')
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
        self.lastest_params = get_dict_params_from(global_model)
    def run(self):
        self.global_model = self.global_model.cuda() # params should be in cuda before aggerating
        rounds = self.options['rounds']
        logging.info(f"server communicates with clients[nums={len(self.clients)},rounds={rounds}].")
        for i in range(rounds):
            # test global model and test client model 
            if i % self.options['eval_round_nums'] == 0:
                loss_recorder, acc_recorder, _ = self.global_trainer.self_test(self.test_loader)
                logging.info("global model test, loss=%.4f, acc=%.4f."%(loss_recorder.avg, acc_recorder.avg) )
                writer.add_scalars(f'global_model_acc',{
                        'test_acc' :  acc_recorder.avg, 
                    }, i)
                writer.add_scalars(f'global_model_loss',{
                        'tess_loss' : loss_recorder.avg,
                    }, i)
            '''
                训练一次的本地客户端，测试的本地数据，先不测
                loss_from_clients = []
                acc_from_clients = []
                for client_id, client in enumerate(self.clients):
                    loss, acc, _ = client.test()
                    loss_from_clients.append(loss.avg)
                    acc_from_clients.append(acc.avg)
                    logging.info("client_%d model test, loss=%.4f, acc=%.4f", client_id, loss.avg, acc.avg)
                logging.info("client models test, loss_avg=%.4f, acc_avg=%.4f."
                             %(sum(loss_from_clients)/len(self.clients), sum(acc_from_clients)/len(self.clients) ))
            '''
            
            # client train and return params
            collect_params = []
            selected_clients = self.select_clients()
            for client in selected_clients:
                logging.info(f"client {client.id} starts to train.")
                client.set_dict_model_params(self.lastest_params)
                client.train()
                # client.test()
                num_samples, params = client.get_data_nums(), client.get_dict_params_from()
                collect_params.append((num_samples, params))
            # optim aggerate params and return new params
            self.lastest_params = self.optim.aggerate(self.global_model, collect_params)
            # update client model
            set_dict_params_to(self.global_model, self.lastest_params)
            '''
            params1 = get_flat_params_from(model_1)
            params2 = get_flat_params_from(model_2)
            print(model_1.state_dict()['model.layer4.1.bn2.num_batches_tracked'])
            print(model_2.state_dict()['model.layer4.1.bn2.num_batches_tracked'])
            print(params1==params2)
            '''
    # return selected clients
    def select_clients(self):
        frac = self.options['client_frac']
        select_nums = max(int(frac*len(self.clients)), 1)
        select_idx = np.random.choice(range(len(self.clients)), select_nums, replace=False)
        clients = [self.clients[i] for i in select_idx]
        return clients
             