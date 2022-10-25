#!/usr/bin/bash

python main_hete.py --optimizer 'fedhete' \
            --client_frac 0.02 \
            --dataset 'cifar10' \
            --part_method 'iid' \
            --rounds 20 \
            --client_nums 100 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_cifar' \
            --lr 0.01 \
            --lr_decay 1 \
            --seed 3 \
            --logdir './fedall_logs/trash_hole'
