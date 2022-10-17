#!/usr/bin/bash

# python main.py --optimizer 'fedall_prox' \
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
#             --fedprox_mu 0.01\
#             --logdir './fedall_logs/fmnist/non_iid/rounds_2000/fedavg/fedall_prox_seed_1-3_rounds_200'

# 下面是prox的方式
python main.py --optimizer 'fedall_prox' \
            --client_nums 100 \
            --client_frac 0.1 \
            --dataset 'fmnist' \
            --part_method 'iid' \
            --rounds 200 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_fedall' \
            --model_source  \
                './fedall_logs/fmnist/iid/rounds_2000/fedavg/seed_1/0200_best_model_acc89.310.stat' \
                './fedall_logs/fmnist/iid/rounds_2000/fedavg/seed_2/0200_best_model_acc88.750.stat' \
                './fedall_logs/fmnist/iid/rounds_2000/fedavg/seed_3/0200_best_model_acc89.000.stat' \
            --model_type 'nn_mnist' 'nn_mnist' 'nn_mnist' \
            --lr 0.01 \
            --lr_decay 1 \
            --fedprox_mu 0.01\
            --logdir './fedall_logs/fmnist/iid/rounds_2000/fedavg/fedall_prox_seed_1-3_rounds_200'

