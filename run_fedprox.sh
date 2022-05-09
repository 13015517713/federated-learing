#!/usr/bin/bash

python main.py --optimizer 'fedprox' \
            --client_nums 100 \
            --client_frac 0.1 \
            --dataset 'mnist' \
            --part_method 'iid' \
            --rounds 10 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_mnist' \
            --lr 0.01
            
