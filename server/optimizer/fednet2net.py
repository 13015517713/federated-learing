import logging
import numpy as np
from server.optimizer.fedbase import BaseServer
from util.model_util import set_dict_params_to
class Server(BaseServer):
    def __init__(self, global_model, global_trainer, global_testset, clients, options):
        super().__init__(global_model, global_trainer, global_testset, clients, options)
        self.options = options

    def run(self):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(self.options['logdir'])
        self.global_model = self.global_model.cuda() # params should be in cuda before aggerating
        rounds = self.options['rounds']
        logging.info(f"server communicates with clients[nums={len(self.clients)},rounds={rounds}].")

        wider_nums = int(self.options['wider_frac']*len(self.clients))
        wider_idx = set(np.random.choice(range(len(self.clients)), wider_nums, replace=False))
        
        for i in range(rounds):
            # test global model and test client model 
            if i % self.options['eval_round_nums'] == 0:
                loss_recorder, acc_recorder, _ = self.global_trainer.self_test(self.test_loader)
                logging.info("global model test, loss=%.4f, acc=%.4f."%(loss_recorder.avg, acc_recorder.avg) )
                writer.add_scalar('gloabl_acc', acc_recorder.avg, i+1)
                writer.add_scalar('global_loss', loss_recorder.avg, i+1)
            
            # select clients train
            selected_clients = self.select_clients()
            collect_params = []
        
            for client in selected_clients:
                # logging.info(f"client {client.id} starts to train.")
                client.set_model_params(self.lastest_params)
                if client.id in wider_idx: client.model.wider()
                client.train()
                if client.id in wider_idx: client.model.dewider()
                wider_nums -= 1
                num_samples, params = client.get_data_nums(), client.get_model_params()
                collect_params.append((num_samples, params))
            self.lastest_params = self.aggerate(collect_params)
            set_dict_params_to(self.global_model, self.lastest_params)