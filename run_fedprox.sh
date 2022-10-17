#!/usr/bin/bash
# bash run_fedprox.sh 1 ./tensorboard_logs/fedprox_non_iid/fedprox_mu1_C2
python main.py --optimizer 'fedprox' \
            --client_nums 100 \
            --client_frac 0.1 \
            --dataset 'cifar10' \
            --part_method 'cifar_non_iid' \
            --rounds 200 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_cifar' \
            --lr 0.01\
            --lr_decay 1 \
            --fedprox_mu $1\
            --logdir './fedall_logs/cifar/non_iid/fedprox_mu0.01'
            
