import os
import argparse
import importlib
import sys
import torch
import numpy as np
import logging
import random
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
from data.get_data import get_dataset_fed
from trainers.client import Client
from servers.server import Server

# GLOBAL PARAMETERS
k_optimizer = ['fedavg', 'fedprox']
k_dataset = ['mnist', 'cifar10']
k_model = ['resnet']
class_num = {
    'cifar10' : 10,
    'mnist' : 10
}
#   return
##      clients : one trainer and dataset per client
##      server  : aggerator
def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer',
                        type=str,
                        choices=k_optimizer,
                        default='fedavg')
    parser.add_argument('--dataset',
                        type=str,
                        choices=k_dataset,
                        default='cifar10')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--epochs',
                        type=int,
                        default=20)
    parser.add_argument('--rounds',
                        type=int,
                        default=50)
    parser.add_argument('--eval_round_nums',
                        type=int,
                        default=1)
    parser.add_argument('--part_method',
                        type=str,
                        default='non-iid')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1)
    parser.add_argument('--model',
                        type=str,
                        choices=k_model,
                        default='resnet')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001)
    # fast:every client has its own memory for model, normal:share one memory space for model
    parser.add_argument('--mode',
                        type=str,
                        choices=['fast','normal'],
                        default='fast')
    parser.add_argument('--client_nums',
                        type=int,
                        default=2)
    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    # set seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    # prepare dataset
    main_train_dataset, main_test_dataset,  \
                clients_trainset_list, clients_testset_list=  \
                    get_dataset_fed(parsed['dataset'], class_num[parsed['dataset']], 
                                    parsed['client_nums'], parsed['part_method'], parsed['alpha'])
    # create model
    model_path = f'trainers.models.%s' % (parsed['model'])
    model_lib = importlib.import_module(model_path)
    model_class = getattr(model_lib, 'Model')
    model_trainer = getattr(model_lib, 'Trainer')
    global_model = model_class(output_features=class_num[parsed['dataset']])
    global_trainer = model_trainer(global_model, epochs=parsed['epochs'], lr=parsed['lr']) # just for test
    # create clients
    clients = []
    for i in range(parsed['client_nums']):
        # model = model_class() if parsed['mode'] == 'fast' else global_model
        model = global_model
        trainer = model_trainer(model, epochs=parsed['epochs'], lr=parsed['lr'])
        client = Client(i, model, trainer, clients_trainset_list[i], 
                        clients_testset_list[i],
                        batch_size=parsed['batch_size'])
        clients.append(client)
    # create optimizer in server
    optim_path = f'servers.optimizers.%s' % (parsed['optimizer'])
    optim_lib = importlib.import_module(optim_path)
    optim_class = getattr(optim_lib, 'Optimizer')
    global_server = Server(global_model, global_trainer, main_test_dataset, clients, optim_class(), parsed)
    logging.debug('server starts to communicate with clients.')
    return global_server

def main():
    global_server = read_options()
    global_server.run()

if __name__ == '__main__':
    main()