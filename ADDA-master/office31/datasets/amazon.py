"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
from data_list import ImageList
import params


def get_amazon(train):

    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize([50,50]),
                                      transforms.RandomResizedCrop(32),  
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=params.dataset_mean,std=params.dataset_std)
                                    #   transforms.RandomResizedCrop(28),                                      
                                      ])

    if train:
        # source_list = '/data/mnist/mnist_train.txt'
        # mnist_data_loader = torch.utils.data.DataLoader(
        #     ImageList(open(source_list).readlines(), transform=pre_process, mode='L'),
        #     batch_size=params.batch_size, shuffle=True, num_workers=0, drop_last=True)
        dataset_source = datasets.ImageFolder(root='/data/office/domain_adaptation_images/amazon/images', transform=pre_process)
        amazon_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=0)
    else:
        # test_list = '/data/mnist/mnist_test.txt'
        # mnist_data_loader = torch.utils.data.DataLoader(
        #     ImageList(open(test_list).readlines(), transform=pre_process, mode='L'),
        #     batch_size=params.batch_size, shuffle=True, num_workers=0, drop_last=True)
        dataset_source = datasets.ImageFolder(root='/data/office/domain_adaptation_images/amazon/images', transform=pre_process)
        amazon_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_source,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=0)
    return amazon_data_loader
