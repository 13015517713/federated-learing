import argparse

k_optimizer = ['fedavg', 'fedprox', 'fednet2net', 'fedall', 'fedall_one_shot', 'fedall_prox']
k_dataset = ['cifar10', 'cifar100', 'mnist', 'fmnist', 'shakespeare']
k_model = ['resnet', 'nn_mnist', 'nn_cifar', 'nn_fedall', 'nn_fedall_80', 'nn_fedall_average', \
            'nn_fedall_max', 'nn_fedall_identity', 'nn_shakespeare']

def args_parse():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()
    # optimizer 
    parser.add_argument('--optimizer',
                        type=str,
                        choices=k_optimizer,
                        default='fedprox')
    parser.add_argument('--fedprox_mu',
                        type=float,
                        default=1)
    # dataset
    parser.add_argument('--dataset',
                        type=str,
                        choices=k_dataset,
                        default='mnist')
    parser.add_argument('--part_method',
                        type=str,
                        default='iid')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1)
    # gloabl configuration
    parser.add_argument('--seed',
                        type=int,
                        default=0)
    parser.add_argument('--rounds',
                        type=int,
                        default=200)
    parser.add_argument('--client_nums',
                        type=int,
                        default=100)
    parser.add_argument('--client_frac',
                        type=float,
                        default=0.1)
    parser.add_argument('--wider_frac',
                        type=float,
                        default=0.3)
    parser.add_argument('--eval_round_nums',
                        type=int,
                        default=1)
    parser.add_argument('--logdir',
                        type=str,
                        default='./tensorboard_logs/trash_hole')
    # local configuration
    parser.add_argument('--batch_size',
                        type=int,
                        default=10)
    parser.add_argument('--epochs',
                        type=int,
                        default=10)
    parser.add_argument('--model',
                        type=str,
                        choices=k_model,
                        default='nn_mnist')
    parser.add_argument('--model_source',
                        nargs='+')
    parser.add_argument('--model_type',
                        nargs='+')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01)
    parser.add_argument('--lr_decay',
                        type=float,
                        default=1)
    
    try: options = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return options