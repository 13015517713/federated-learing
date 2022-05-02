import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
k_optimizer = ['fedavg']
k_dataset = ['cifar10', 'mnist']
k_model = ['resnet', 'nn_mnist']
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
                        default='mnist')
    parser.add_argument('--batch_size',
                        type=int,
                        default=10)
    parser.add_argument('--epochs',
                        type=int,
                        default=10)
    parser.add_argument('--rounds',
                        type=int,
                        default=100)
    # test global model for global test_set per eval_round_nums
    parser.add_argument('--eval_round_nums',
                        type=int,
                        default=1)
    parser.add_argument('--part_method',
                        type=str,
                        default='mnist_non_iid')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1)
    parser.add_argument('--model',
                        type=str,
                        choices=k_model,
                        default='nn_mnist')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01)
    parser.add_argument('--client_nums',
                        type=int,
                        default=100)
    parser.add_argument('--client_frac',
                        type=float,
                        default=0.1)
    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    # set seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    # prepare dataset
    _, main_test_dataset,  \
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
        model = model_class(output_features=class_num[parsed['dataset']])
        trainer = model_trainer(model, epochs=parsed['epochs'], lr=parsed['lr'])
        client = Client(i, model, trainer, clients_trainset_list[i], 
                        clients_testset_list[i],
                        # main_test_dataset,
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