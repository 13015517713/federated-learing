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
load_method = {
    'cifar10' : load_cifar10_data
}
def get_dataset_fed(data_name, class_nums, client_nums, method, alpha=None, data_home=kstored_path):
    train_dataset, test_dataset = load_method[data_name]()
    train_len = len(train_dataset)
    train_y = np.array(train_dataset.targets)
    data_idx = [[] for _ in range(client_nums)]
    if method == "iid":
        idxs = np.random.permutation(train_len)
        data_idx = np.array_split(idxs, client_nums)
    elif method == "non-iid":
        class_distribution = np.random.dirichlet([alpha]*client_nums, class_nums)
        class_idx = [np.where(train_y==i)[0] for i in range(class_nums)]
        client_idx = [[] for _ in range(client_nums)]
        for c, frac in zip(class_idx, class_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(frac)[:-1]*len(c)).astype(int) )):
                data_idx[i] += idcs.tolist()
    # set one class for every client
    elif method == "non-iid-one":
        assert class_nums==client_nums
        for client_id in range(client_nums):
            onedata_idx = np.where(train_y == client_id)[0]
            data_idx[client_id] = onedata_idx
    all_len = len(train_y)
    client_train_set = []
    client_test_set = []
    for i, idx in enumerate(data_idx):
        # split client data to train and test
        np.random.shuffle(idx)
        train_len = 0.9*len(idx)
        train_idx = idx[:train_len]
        test_idx = idx[train_len:]
        client_train_set.append(Subset(train_dataset, train_idx))
        client_test_set.append(Subset(train_dataset, test_idx))
        print(f'client id={i} train_set length is {len(idx)}.')
    print(f'all train_set length is {all_len}.')
    return train_dataset, test_dataset, client_train_set, client_test_set
    
if __name__ == '__main__':
    get_dataset_fed('cifar10', 10, 16, 'non-iid', alpha=0.1)

    