#!/usr/bin/bash

# python main.py --optimizer 'fedavg' \
#             --client_nums 10 \
#             --client_frac 0.3 \
#             --dataset 'cifar100' \
#             --part_method 'iid' \
#             --rounds 200 \
#             --epochs 10 \
#             --batch_size 10 \
#             --model 'nn_cifar' \
#             --lr 0.01 \
#             --lr_decay 1 \
#             --seed 3 \
#             --logdir './fedall_logs/trash_hole'

# 运行固定数据
python main.py --optimizer 'fedavg' \
            --client_frac 0.1 \
            --dataset 'cifar10' \
            --rounds 200 \
            --client_nums 10 \
            --epochs 2 \
            --batch_size 10 \
            --model 'nn_cifar' \
            --lr 0.01 \
            --lr_decay 1 \
            --seed 3 \
            --logdir './fedall_logs/trash_hole'
