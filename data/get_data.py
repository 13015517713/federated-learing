from matplotlib.pyplot import axis
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
def load_mnist_data(datahome=kstored_path):
    trans_func = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(datahome, train=True, transform=trans_func, download=True)
    test_dataset = torchvision.datasets.MNIST(datahome, train=False, transform=trans_func, download=True)
    return train_dataset, test_dataset
load_method = {
    'cifar10' : load_cifar10_data,
    'mnist' : load_mnist_data
}
def get_dataset_fed(data_name, class_nums, client_nums, method, alpha=None, data_home=kstored_path):
    train_dataset, test_dataset = load_method[data_name]()
    train_y = np.array(train_dataset.targets)
    data_idx = [[] for _ in range(client_nums)]
    if method == "iid":
        data_idx = common_iid(train_y, client_nums)   
    elif method == "non-iid-dirichlet":
        data_idx = non_iid_dirichlet(train_y, class_nums, client_nums, alpha)
    elif method == "mnist_non_iid":
        # dataset splits to 200*300, 300 belongs to one class,  
        data_idx = mnist_non_iid(train_y, client_nums)
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
    num_shards, num_imgs = 100, 600
    idx_shard = [i for i in range(num_shards)]
    data_idx = [[] for i in range(client_num)]
    idxs = np.arange(num_shards*num_imgs)
    # sort labels
    idxs_labels = np.vstack((idxs, train_y))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign 2 shards/client
    for i in range(client_num):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
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

if __name__ == '__main__':
    get_dataset_fed('cifar10', 10, 16, 'non-iid', alpha=0.1)

    