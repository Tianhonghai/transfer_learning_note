import dann
import random
import PIL
import os
from data_list import ImageList
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# print(sys.path)

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
import models
from models.model import CNNModel
from models import model
import numpy as np
from test import test

# task, n_label = 'M->U', 10
# task, n_label = 'U->M', 10
# task, n_label = 'M->S', 10
# task, n_label = 'S->M', 10
# task, n_label = 'A->W', 31
task, n_label = 'Ar->Cl', 65

image_size= 28
# source_dataset_name = 'MNIST'
# source_dataset_name = 'USPS'
# source_dataset_name = 'SVHN'

# target_dataset_name = 'MNIST'
# target_dataset_name = 'USPS'
# target_dataset_name = 'SVHN'

# source_image_root = os.path.join('..', 'dataset', source_dataset_name)
# target_image_root = os.path.join('..', 'dataset', target_dataset_name)
# model_root = os.path.join('..', 'models')


# load data

batch_size = 128

if task == 'M->U':
    # img_transform_source = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])
    # img_transform_target = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])    
    # dataset_source = datasets.MNIST(
    #     root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset')),
    #     train=True,
    #     transform=img_transform_source,
    #     download=True
    # )
    # dataset_target = datasets.USPS(
    #     root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset')),
    #     train=True,
    #     transform=img_transform_target,
    #     download=True
    # )
    # dataloader_source = torch.utils.data.DataLoader(
    #     dataset=dataset_source,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=8)
    # dataloader_target = torch.utils.data.DataLoader(
    #     dataset=dataset_target,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=8)

    source_list = '/data/mnist/mnist_train.txt'
    target_list = '/data/usps/usps_train.txt'
    test_list = '/data/usps/usps_test.txt'

    dataloader_source = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_target = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dann.dann(dataloader_source, dataloader_target,image_size,batch_size,'MNIST', 'USPS',n_label)

if task == 'U->M':

    source_list = '/data/usps/usps_train.txt'
    target_list = '/data/mnist/mnist_train.txt'
    test_list = '/data/mnist/mnist_test.txt'

    dataloader_source = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_target = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dann.dann(dataloader_source, dataloader_target,image_size,batch_size,'USPS', 'MNIST',n_label)    

if task == 'M->S':
    source_list = '/data/mnist/mnist_train.txt'
    target_list = '/data/svhn/svhn_train.txt'
    test_list = '/data/svhn/svhn_test.txt'

    dataloader_source = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_target = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dann.dann(dataloader_source, dataloader_target,image_size,batch_size,'MNIST', 'SVHN',n_label)

if task == 'S->M':
    source_list = '/data/svhn/svhn_train.txt'
    target_list = '/data/mnist/mnist_train.txt'
    test_list = '/data/mnist/mnist_test.txt'

    dataloader_source = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_target = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=batch_size, shuffle=True, num_workers=1)
    dann.dann(dataloader_source, dataloader_target,image_size,batch_size,'SVHN', 'MNIST',n_label)

if task == 'A->W':
    img_transform_source = transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BILINEAR),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Compose(
    #     <pre_process.ResizeImage object at 0x7f4c83272f90>
    #     RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BILINEAR)
    #     RandomHorizontalFlip(p=0.5)
    #     ToTensor()
    #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # )

    dataset_source = datasets.ImageFolder(root='/data/office/domain_adaptation_images/amazon/images', transform=img_transform_source)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    dataset_target = datasets.ImageFolder(root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset','Original_images','webcam','images')), transform=img_transform_source)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    dann.dann(dataloader_source, dataloader_target,image_size,batch_size,'amazon', 'webcam',n_label)    

if task == 'Ar->Cl':
    # train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    # dataset_target = GetLoader(
    #     data_root=os.path.join(target_image_root, 'train'),
    #     data_list=train_list,
    #     transform=img_transform_target
    # )
    img_transform_source = transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset_source = datasets.ImageFolder(root='/data/office-home/images/Art', transform=img_transform_source)
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    dataset_target = datasets.ImageFolder(root='/data/office-home/images/Clipart', transform=img_transform_source)
    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
    dann.dann(dataloader_source, dataloader_target,image_size,batch_size,'Art', 'Clipart',n_label)