#!/usr/bin/bash

python main.py --optimizer 'fedavg' \
            --client_nums 10 \
            --client_frac 0.3 \
            --dataset 'cifar10' \
            --part_method 'cifar_non_iid' \
            --rounds 2000 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_cifar' \
            --lr 0.01 \
            --lr_decay 1 \
            --logdir './tensorboard_logs/fedall/cifar_2000r/fedavg'
