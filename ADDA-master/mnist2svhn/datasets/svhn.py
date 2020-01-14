"""Dataset setting and data loader for svhn.

Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_svhn.py
"""

import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import params
from data_list import ImageList

def get_svhn(train):

    # image pre-processing
    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                       mean=params.dataset_mean,
    #                                       std=params.dataset_std)])

    # # dataset and data loader
    # svhn_dataset = svhn(root=params.data_root,
    #                     train=train,
    #                     transform=pre_process,
    #                     download=True)

    # svhn_data_loader = torch.utils.data.DataLoader(
    #     dataset=svhn_dataset,
    #     batch_size=params.batch_size,
    #     shuffle=True)

    # return svhn_data_loader

    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    if train:
        source_list = '/data/svhn/svhn_train.txt'
        svhn_data_loader = torch.utils.data.DataLoader(
            ImageList(open(source_list).readlines(), transform=pre_process, mode='L'),
            batch_size=params.batch_size, shuffle=True, num_workers=0, drop_last=True)
    else:
        test_list = '/data/svhn/svhn_test.txt'
        svhn_data_loader = torch.utils.data.DataLoader(
            ImageList(open(test_list).readlines(), transform=pre_process, mode='L'),
            batch_size=params.batch_size, shuffle=True, num_workers=0, drop_last=True)
    return svhn_data_loader