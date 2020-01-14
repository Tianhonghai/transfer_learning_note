"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
from data_list import ImageList
import params


def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),                                      
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    if train:
        source_list = '/data/mnist/mnist_train.txt'
        mnist_data_loader = torch.utils.data.DataLoader(
            ImageList(open(source_list).readlines(), transform=pre_process, mode='L'),
            batch_size=params.batch_size, shuffle=True, num_workers=0, drop_last=True)
    else:
        test_list = '/data/mnist/mnist_test.txt'
        mnist_data_loader = torch.utils.data.DataLoader(
            ImageList(open(test_list).readlines(), transform=pre_process, mode='L'),
            batch_size=params.batch_size, shuffle=True, num_workers=0, drop_last=True)
    return mnist_data_loader
