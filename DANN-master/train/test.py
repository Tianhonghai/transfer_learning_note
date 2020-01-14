import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader import GetLoader
from torchvision import datasets
import PIL
from data_list import ImageList

def test(dataset_name, epoch,image_size):
    # assert dataset_name in ['MNIST', 'mnist_m']
    assert dataset_name in ['MNIST', 'USPS','SVHN','Art','Clipart','amazon','webcam']

    # model_root = os.path.join('..', 'models')
    model_root = os.path.join(os.path.dirname(__file__), os.pardir,'models')
    image_root = os.path.join('..', 'dataset', dataset_name)

    cuda = False
    cudnn.benchmark = True
    batch_size = 128
    # image_size = 28

    alpha = 0

    """load data"""

    # img_transform_source = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])
    img_transform_target = transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # img_transform_target = transforms.Compose([
    #     transforms.Resize([image_size,image_size]),
    #     transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BILINEAR),
    #     # transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # if dataset_name == 'mnist_m':
    #     test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
    #     dataset = GetLoader(
    #         data_root=os.path.join(image_root, 'mnist_m_test'),
    #         data_list=test_list,
    #         transform=img_transform_target
    #     )
    # if dataset_name == 'USPS':
    #     dataset = datasets.USPS(
    #         root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset')),
    #         train=False,
    #         transform=img_transform_source,
    #         download=True
    #     )
    # elif dataset_name == 'MNIST':
    #     dataset = datasets.MNIST(
    #         root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset')),
    #         train=False,
    #         transform=img_transform_source,
    #         download=True
    #     )
    # elif dataset_name == 'SVHN':
    #     dataset = datasets.SVHN(
    #         root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset')),
    #         split='test',
    #         transform=img_transform_target,
    #         download=True
    #     ) 
    # elif dataset_name == 'Art':
    #     dataset = datasets.ImageFolder(root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset','OfficeHomeDataset_10072016','Art')), transform=img_transform_target)
    # elif dataset_name == 'Clipart':
    #     dataset = datasets.ImageFolder(root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset','OfficeHomeDataset_10072016','Clipart')), transform=img_transform_target)
    # elif dataset_name == 'amazon':
    #     dataset = datasets.ImageFolder(root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset','Original_images','amazon','images')), transform=img_transform_target)
    # elif dataset_name == 'webcam':
    #     dataset = datasets.ImageFolder(root=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,'dataset','Original_images','webcam','images')), transform=img_transform_target)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=8
    # )

    if dataset_name == 'USPS':
        list = '/data/usps/usps_test.txt'
        dataloader = torch.utils.data.DataLoader(
            ImageList(open(list).readlines(), transform=transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]), mode='RGB'),
            batch_size=batch_size, shuffle=True, num_workers=1)          
    elif dataset_name == 'MNIST':
        list = '/data/mnist/mnist_test.txt'
        dataloader = torch.utils.data.DataLoader(
            ImageList(open(list).readlines(), transform=transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]), mode='RGB'),
            batch_size=batch_size, shuffle=True, num_workers=1)        
    elif dataset_name == 'SVHN':
        list = '/data/svhn/svhn_test.txt'
        dataloader = torch.utils.data.DataLoader(
            ImageList(open(list).readlines(), transform=transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]), mode='RGB'),
            batch_size=batch_size, shuffle=True, num_workers=1)
    if dataset_name == 'Art':
        dataset = datasets.ImageFolder(root='/data/office-home/images/Art', transform=img_transform_target)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )
    elif dataset_name == 'Clipart':
        dataset = datasets.ImageFolder(root='/data/office-home/images/Clipart', transform=img_transform_target)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )
    elif dataset_name == 'amazon':
        dataset = datasets.ImageFolder(root='/data/office/domain_adaptation_images/amazon/images', transform=img_transform_target)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )        
    elif dataset_name == 'webcam':
        dataset = datasets.ImageFolder(root='/data/office/domain_adaptation_images/webcam/images', transform=img_transform_target)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)

        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
