#!/usr/bin/bash

# 运行cifar
python main.py --optimizer 'fedall' \
            --client_nums 100 \
            --client_frac 0.1 \
            --dataset 'cifar10' \
            --part_method 'iid' \
            --rounds 200 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_fedall' \
            --model_source  \
                './fedall_logs/cifar/iid/rounds_200/seed_1/fedavg/model.stat' \
                './fedall_logs/cifar/iid/rounds_200/seed_2/fedavg/model.stat' \
                './fedall_logs/cifar/iid/rounds_200/seed_3/fedavg/model.stat' \
            --model_type 'nn_cifar' 'nn_cifar' 'nn_cifar' \
            --lr 0.01 \
            --lr_decay 1 \
            --logdir './fedall_logs/trash_hole'

# 运行fmnist
# python main.py --optimizer 'fedall' \
#             --client_nums 100 \
#             --client_frac 0.1 \
#             --dataset 'fmnist' \
#             --part_method 'fmnist_non_iid' \
#             --rounds 200 \
#             --epochs 10 \
#             --batch_size 10 \
#             --model 'nn_fedall' \
#             --model_source  \
#                 './fedall_logs/fmnist/non_iid/rounds_2000/fedavg/seed_1/0200_best_model_acc84.760.stat' \
#                 './fedall_logs/fmnist/non_iid/rounds_2000/fedavg/seed_2/0200_best_model_acc84.670.stat' \
#                 './fedall_logs/fmnist/non_iid/rounds_2000/fedavg/seed_3/0200_best_model_acc84.110.stat' \
#             --model_type 'nn_mnist' 'nn_mnist' 'nn_mnist' \
#             --lr 0.01 \
#             --lr_decay 1 \
#             --logdir './fedall_logs/fmnist/non_iid/rounds_2000/fedavg/fedall_seed_1-3_rounds_200'

# 下面是合并shakespeare
# python main.py --optimizer 'fedall' \
#             --client_frac 0.1 \
#             --dataset 'shakespeare' \
#             --rounds 60 \
#             --epochs 10 \
#             --batch_size 10 \
#             --model 'nn_fedall_80' \
#             --model_source  \
#                 './fedall_logs/shakespeare/fedavg/seed_1/0060_best_model_acc49.020.stat' \
#                 './fedall_logs/shakespeare/fedavg/seed_2/0060_best_model_acc48.969.stat' \
#                 './fedall_logs/shakespeare/fedavg/seed_3/0060_best_model_acc49.053.stat' \
#             --model_type 'nn_shakespeare' 'nn_shakespeare' 'nn_shakespeare' \
#             --lr 0.01 \
#             --lr_decay 1 \
#             --logdir './fedall_logs/trash_hole'