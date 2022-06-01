#!/usr/bin/bash

python main.py --optimizer 'fedall' \
            --client_nums 10 \
            --client_frac 0.3 \
            --dataset 'cifar10' \
            --part_method 'cifar_non_iid' \
            --rounds 200 \
            --epochs 10 \
            --batch_size 10 \
            --model 'nn_fedall' \
            --model_source './tensorboard_logs/trash_hole/fedprox/model.stat' \
                    './tensorboard_logs/trash_hole/model.stat' \
            --model_type 'nn_cifar' 'nn_cifar' \
            --lr 0.01 \
            --lr_decay 1 \
            --logdir './tensorboard_logs/trash_hole/fedall'