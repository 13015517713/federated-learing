import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import importlib
import torch
import numpy as np
import logging
import random
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
from data.get_data import get_dataset_fed
from local.client import Client
from util.options import args_parse

class_num = {
    'cifar10' : 10,
    'mnist' : 10
}

def get_debug():
    import debugpy
    import setproctitle
    setproctitle.setproctitle("fednet2net")
    debugpy.listen(10000)
    debugpy.wait_for_client()

def read_options():
    options = args_parse()
    print(options)
    
    # set seed
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    
    # prepare dataset
    _, main_test_dataset,  \
        clients_trainset_list, clients_testset_list=  \
            get_dataset_fed(options['dataset'], class_num[options['dataset']], 
                options['client_nums'], options['part_method'], options['alpha'])

    # create model
    model_path = f'local.model.%s' % (options['model'])
    model_lib = importlib.import_module(model_path)
    model_class = getattr(model_lib, 'Model')
    global_model = model_class(output_features=class_num[options['dataset']])
    
    # create trainer
    trainer_path = f'local.trainer.%s' % ('trainer_fedprox' 
                                    if options['optimizer']=='fedprox' else 'trainer_common')
    trainer_lib = importlib.import_module(trainer_path)
    model_trainer = getattr(trainer_lib, 'Trainer')
    global_trainer = model_trainer(global_model, options=options) # just for test
    
    # create clients
    clients = []
    for i in range(options['client_nums']):
        model = model_class(output_features=class_num[options['dataset']])
        trainer = model_trainer(model, options)
        client = Client(i, model, trainer, clients_trainset_list[i], 
                        clients_testset_list[i],
                        # main_test_dataset,
                        batch_size=options['batch_size'])
        clients.append(client)
        
    # create optimizer in server
    optim_path = f'server.optimizer.%s' % (options['optimizer'])
    optim_lib = importlib.import_module(optim_path)
    optim_class = getattr(optim_lib, 'Server')
    global_server = optim_class(global_model, global_trainer, main_test_dataset, clients, options)
    logging.debug('server starts to communicate with clients.')
    return global_server

def main():
    global_server = read_options()
    global_server.run()
    
if __name__ == '__main__':
    # get_debug()
    main()