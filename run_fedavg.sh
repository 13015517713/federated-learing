#!/usr/bin/bash

python main.py --optimizer 'fedavg' \
            --client_nums 100 \
            --client_frac 0.1 \
            --dataset 'mnist' \
            --part_method 'mnist_non_iid' \
            --rounds 20 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_mnist' \
            --lr 0.01 \
            --lr_decay 1 \
            --logdir './tensorboard_logs/fedall/fedavg/C2'
            
