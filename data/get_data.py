import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Subset 
# kstored_path = './stored_data'
kstored_path = '/home/wcx/gitProject/federated-learing/data/stored_data'
def cifar10_transforms():
    mean=[0.49139968,0.48215841,0.44653091]
    stdv=[ 0.2023,0.1994,0.2010]  
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    return train_transforms, test_transforms
#   load cifar10 data
#   return train dataset and test dataset
##      train_dataset : include train_X and train_Y
##      test_dataset : include test_X and test_Y
def load_cifar10_data(datahome=kstored_path):
    train_tran, test_tran = cifar10_transforms()
    train_dataset = torchvision.datasets.CIFAR10(datahome, train=True, transform=train_tran, download=True)
    test_dataset = torchvision.datasets.CIFAR10(datahome, train=False, transform=test_tran)
    return train_dataset, test_dataset
load_method = {
    'cifar10' : load_cifar10_data
}
#   get non iid dataset
##      test dataset is global, not local dataset splited
##      in this situation, all clients share one dataset
#    return
##     maintrain_set : full trainset
##     maintest_set : full testset
##     [subtrain_set] : splited train_dataset for every client
##     [subtest_set] : full testset
#    ref
##     
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
    all_len = 0
    for i, data in enumerate(data_idx):
        all_len += len(data)
        print(f'client id={i} train_set length is {len(data)}.')
    print(f'all train_set length is {all_len}.')
    # create sub dataset
    subtrain_set = [Subset(train_dataset, data_idx[i]) for i in range(client_nums)] 
    subtest_set = [test_dataset for _ in range(client_nums)]
    return train_dataset, test_dataset, subtrain_set, subtest_set
    
if __name__ == '__main__':
    get_dataset_fed('cifar10', 10, 16, 'non-iid', alpha=0.1)

    