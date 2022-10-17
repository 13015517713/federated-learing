import glob
import json
import tqdm
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Subset

kstored_path = 'data/stored_data'
def load_cifar10_data(datahome=kstored_path):
    trans_func = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(datahome, train=True, transform=trans_func, download=True)
    test_dataset = torchvision.datasets.CIFAR10(datahome, train=False, transform=trans_func, download=True)
    return train_dataset, test_dataset
def load_cifar100_data(datahome=kstored_path):
    trans_func = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR100(datahome, train=True, transform=trans_func, download=True)
    test_dataset = torchvision.datasets.CIFAR100(datahome, train=False, transform=trans_func, download=True)
    return train_dataset, test_dataset
def load_mnist_data(datahome=kstored_path):
    trans_func = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(datahome, train=True, transform=trans_func, download=True)
    test_dataset = torchvision.datasets.MNIST(datahome, train=False, transform=trans_func, download=True)
    return train_dataset, test_dataset
def load_fmnist_data(datahome=kstored_path):
    trans_func = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(datahome, train=True, transform=trans_func, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(datahome, train=False, transform=trans_func, download=True)
    return train_dataset, test_dataset
load_method = {
    'cifar10' : load_cifar10_data,
    'cifar100' : load_cifar100_data,
    'mnist' : load_mnist_data,
    'fmnist' : load_fmnist_data
}
def get_dataset_fed(data_name, class_nums, client_nums, method, alpha=None, data_home=kstored_path):
    train_dataset, test_dataset = load_method[data_name]()
    train_y = np.array(train_dataset.targets)
    data_idx = [[] for _ in range(client_nums)]
    if method == "iid":
        data_idx = common_iid(train_y, client_nums)   
    elif method == "non-iid-dirichlet":
        data_idx = non_iid_dirichlet(train_y, class_nums, client_nums, alpha)
    elif method == "mnist_non_iid" or method == "fmnist_non_iid":
        # dataset splits to 200*300, 300 belongs to one class,  
        data_idx = mnist_non_iid(train_y, client_nums)
    elif method == "cifar_non_iid":
        # dataset splits to 200*300, 300 belongs to one class,  
        data_idx = cifar_non_iid(train_y, client_nums)
    else: 
        # just one class per client
        data_idx = non_iid_one_class(train_y, class_nums, client_nums)
    client_train_set = []
    client_test_set = []
    for i, idx in enumerate(data_idx):
        # split client data to train and test
        np.random.shuffle(idx)
        train_len = int(0.9*len(idx))
        train_idx = idx[:train_len]
        test_idx = idx[train_len:]
        client_train_set.append(Subset(train_dataset, train_idx))
        client_test_set.append(Subset(train_dataset, test_idx))
        # 统计训练集测试集数据分布
        train_per_class_num = [0]*class_nums
        test_per_class_num = [0]*class_nums
        for j in train_idx: train_per_class_num[train_dataset[j][1]]+=1
        for j in test_idx: test_per_class_num[train_dataset[j][1]]+=1 
        print(f'client id={i} train_set length is {len(idx)}.')
        print(f'train class distribution, {train_per_class_num}')
        print(f'test class distribution, {test_per_class_num}')
    print(f'all train_set length is {len(train_y)}.')
    return train_dataset, test_dataset, client_train_set, client_test_set
def common_iid(train_y, client_nums):
    idxs = np.random.permutation(len(train_y))
    return np.array_split(idxs, client_nums)
def non_iid_dirichlet(train_y, class_nums, client_nums, alpha):
    class_distribution = np.random.dirichlet([alpha]*client_nums, class_nums)
    class_idx = [np.where(train_y==i)[0] for i in range(class_nums)]
    data_idx = [[] for _ in range(client_nums)]
    for c, frac in zip(class_idx, class_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(frac)[:-1]*len(c)).astype(int) )):
            data_idx[i] += idcs.tolist()
    return data_idx
def mnist_non_iid(train_y, client_num):
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    data_idx = [[] for i in range(client_num)]
    idxs = np.arange(num_shards*num_imgs)
    # sort labels
    idxs_labels = np.vstack((idxs, train_y))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign 2 shards/client
    for i in range(client_num):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        data_idx[i] = np.concatenate([idxs[rand*num_imgs:(rand+1)*num_imgs]
                                     for rand in rand_set], axis=0)
    return data_idx
def cifar_non_iid(train_y, client_num):
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    data_idx = [[] for i in range(client_num)]
    idxs = np.arange(num_shards*num_imgs)
    # sort labels
    idxs_labels = np.vstack((idxs, train_y))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign 2 shards/client
    for i in range(client_num):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        data_idx[i] = np.concatenate([idxs[rand*num_imgs:(rand+1)*num_imgs]
                                     for rand in rand_set], axis=0)
    return data_idx
def non_iid_one_class(train_y, class_nums, client_nums):
    assert class_nums==client_nums, "client_num isn't equal to class_num in non-iid-one"
    data_idx = [[] for _ in range(client_nums)]
    for client_id in range(client_nums):
        onedata_idx = np.where(train_y == client_id)[0]
        data_idx[client_id] = onedata_idx
    return data_idx


def load_shakespeare_data(data_home):
    train_json = glob.glob(f'{data_home}/data/train/*.json')
    test_json = glob.glob(f'{data_home}/data/test/*.json')

    # convert train_json data to torch dataset
    ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
    NUM_LETTERS = len(ALL_LETTERS) # 80个字符
    
    def get_dataset_from_json(json_path):
        with open(json_path, 'r') as r:
            json_file = json.load(r)
            users_list = json_file['users']
            users_data = json_file['user_data']
        tensor_data = []
        for user_idx, user_str in enumerate(tqdm.tqdm(users_list)):
            # process input
            data_x = users_data[user_str]['x']
            input_indices = []
            for sentence in data_x:
                indices = []
                for c in sentence:
                    indices.append(ALL_LETTERS.find(c))
                input_indices.append(indices)
            input_indices = torch.tensor(input_indices)
        
            #  process target
            data_y = users_data[user_str]['y']
            target_code = []
            for next_char in data_y:
                target_code.append(ALL_LETTERS.find(next_char))
            target_code = torch.tensor(target_code)
            tensor_data.append([input_indices, target_code])

        # generate main_dataset and single dataset list
        client_dataset = [torch.utils.data.TensorDataset(i[0],i[1]) for i in tensor_data]
        main_dataset = torch.utils.data.TensorDataset(
            torch.cat([i[0] for i in tensor_data], dim=0),
            torch.cat([i[1] for i in tensor_data], dim=0),
        )
        return main_dataset, client_dataset
    print('--- start to process trainset ---')
    _, clients_trainset_list = get_dataset_from_json(train_json[0])
    print('--- start to process testset ---')
    main_test_dataset, clients_testset_list = get_dataset_from_json(test_json[0])
    assert len(clients_trainset_list) == len(clients_testset_list)
    return len(clients_trainset_list), _, main_test_dataset, \
                        clients_trainset_list, clients_testset_list
    
load_method_fixed = {
    'shakespeare' : load_shakespeare_data
}

# 返回客户端数量，全局测试集，训练集，客户端训练集，客户端测试集
def get_dataset_fixed(dataname, data_home=kstored_path):
    client_num, _, main_test_dataset, \
        clients_trainset_list, clients_testset_list = \
            load_method_fixed[dataname](f'{data_home}/{dataname}')
    return client_num, _, main_test_dataset, clients_trainset_list, clients_testset_list

if __name__ == '__main__':
    get_dataset_fed('cifar10', 10, 16, 'non-iid', alpha=0.1)

    