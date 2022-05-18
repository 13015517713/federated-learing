#!/usr/bin/bash

python main.py --optimizer 'fednet2net' \
            --client_nums 100 \
            --client_frac 0.1 \
            --dataset 'mnist' \
            --part_method 'mnist_non_iid' \
            --rounds 200 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_mnist' \
            --wider_frac 0.3 \
            --lr 0.01 \
            --lr_decay 1 \
            --logdir './tensorboard_logs/fednet2net/sample_equal/C2_fednet2net'
            
