#!/usr/bin/bash
# test fedall_one_shot average
# python main.py --optimizer 'fedall_one_shot' \
#             --client_nums 10 \
#             --dataset 'cifar10' \
#             --part_method 'iid' \
#             --model 'nn_fedall_average' \
#             --model_source './tensorboard_logs/fedall/cifar_200r/fedavg/model.stat' \
#                     './tensorboard_logs/fedall/cifar_200r/fedprox_mu0.1/model.stat' \
#             --model_type 'nn_cifar' 'nn_cifar'

# test fedall_one_shot max
# python main.py --optimizer 'fedall_one_shot' \
#             --client_nums 10 \
#             --dataset 'cifar10' \
#             --part_method 'iid' \
#             --model 'nn_fedall_max' \
#             --model_source './tensorboard_logs/fedall/cifar_200r/fedavg/model.stat' \
#                     './tensorboard_logs/fedall/cifar_200r/fedprox_mu0.1/model.stat' \
#             --model_type 'nn_cifar' 'nn_cifar'

# test fedall_identity
# python main.py --optimizer 'fedall_one_shot' \
#             --client_nums 100 \
#             --dataset 'cifar10' \
#             --part_method 'iid' \
#             --model 'nn_fedall_identity' \
#             --model_source './tensorboard_logs/fedall/cifar_2000r/fedavg/model.stat' \
#             --model_type 'nn_cifar'

# test for mnist

# python main.py --optimizer 'fedall_one_shot' \
#             --client_nums 1 \
#             --dataset 'mnist' \
#             --part_method 'iid' \
#             --model 'nn_fedall_identity' \
#             --model_source './tensorboard_logs/fedall/fedprox/C2_r2000_mu1/model.stat' \
#             --model_type 'nn_mnist'

# test fedall
# python main.py --optimizer 'fedall_one_shot' \
#             --client_nums 1 \
#             --dataset 'mnist' \
#             --part_method 'iid' \
#             --model 'nn_fedall_max' \
#             --model_source './tensorboard_logs/fedall/fedprox/C2_r2000_mu1/model.stat' \
#                     './tensorboard_logs/fedall/fedavg/C2_r2000/model.stat' \
#             --model_type 'nn_mnist' 'nn_mnist'