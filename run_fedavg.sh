#!/usr/bin/bash

python main.py --optimizer 'fedavg' \
            --client_nums 16 \
            --dataset 'cifar10' \
            --part_method 'non-iid' \
            --alpha 0.1 \
            --rounds 50 \
            --epochs 20 \
            --batch_size 64 \
            --model 'resnet' \
            --lr 0.003
            
