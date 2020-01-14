from torchvision import datasets
import os.path as osp
import os
from PIL import Image
import torch


outdir = '/data/mnist'

mnist_dataset_train = datasets.MNIST('/data/mnist', train=True, transform=None, 
                               target_transform=None,download=True)
mnist_train_image = osp.join(outdir, 'mnist_train_image')
os.system("mkdir -p " + mnist_train_image)
with open(osp.join(outdir, 'mnist_train.txt'), 'w') as label_file:
    for i in range(len(mnist_dataset_train)):
        img = Image.fromarray(mnist_dataset_train.train_data[i].numpy())
        img = img.resize([32,32])
        img = img.convert('RGB')
        img.save(osp.join(mnist_train_image, '{:d}.png'.format(i)))
        label_file.write(mnist_train_image+'/{:d}.png {:d}\n'.format(i, mnist_dataset_train.train_labels[i]))

mnist_dataset_test = datasets.MNIST('/data/mnist', train=False, transform=None, 
                               target_transform=None,download=True)
mnist_test_image = osp.join(outdir, 'mnist_test_image')
os.system("mkdir -p " + mnist_test_image)
with open(osp.join(outdir, 'mnist_test.txt'), 'w') as label_file:
    for i in range(len(mnist_dataset_test)):
        img = Image.fromarray(mnist_dataset_test.test_data[i].numpy())
        img = img.resize([32,32])
        img = img.convert('RGB')
        img.save(osp.join(mnist_test_image, '{:d}.png'.format(i)))
        label_file.write(mnist_test_image+'/{:d}.png {:d}\n'.format(i, mnist_dataset_test.test_labels[i]))

outdir = '/data/svhn'
svhn_dataset_train = datasets.SVHN('/data/svhn', split='train', transform=None, 
                             target_transform=None,download=True)     
svhn_train_image = osp.join(outdir, 'svhn_tain_image')
os.system("mkdir -p " + svhn_train_image)
svhn_labels = svhn_dataset_train.labels.flatten()
with open(osp.join(outdir, 'svhn_train.txt'), 'w') as label_file:
    for i in range(len(svhn_dataset_train)):
        img = Image.fromarray(svhn_dataset_train.data[i].transpose(1,2,0))
        img.save(osp.join(svhn_train_image, '{:d}.png'.format(i)))
        label_file.write(svhn_train_image+'/{:d}.png {:d}\n'.format(i, svhn_labels[i]))

svhn_dataset_test = datasets.SVHN('/data/svhn', split='test', transform=None, 
                             target_transform=None,download=True)     
svhn_test_image = osp.join(outdir, 'svhn_test_image')
os.system("mkdir -p " + svhn_test_image)
svhn_labels = svhn_dataset_test.labels.flatten()
with open(osp.join(outdir, 'svhn_test.txt'), 'w') as label_file:
    for i in range(len(svhn_test_image)):
        img = Image.fromarray(svhn_dataset_test.data[i].transpose(1,2,0))
        img.save(osp.join(svhn_test_image, '{:d}.png'.format(i)))
        label_file.write(svhn_test_image+'/{:d}.png {:d}\n'.format(i, svhn_labels[i]))

outdir = '/data/usps'
usps_dataset_train = datasets.USPS('/data/usps', train=True, transform=None, 
                             target_transform=None,download=True)     
usps_train_image = osp.join(outdir, 'usps_tain_image')
os.system("mkdir -p " + usps_train_image)
usps_labels = usps_dataset_train.targets  # .labels.flatten()
with open(osp.join(outdir, 'usps_train.txt'), 'w') as label_file:
    for i in range(len(usps_dataset_train)):
        img = Image.fromarray(usps_dataset_train.data[i])
        img.save(osp.join(usps_train_image, '{:d}.png'.format(i)))
        label_file.write(usps_train_image+'/{:d}.png {:d}\n'.format(i, usps_labels[i]))

usps_dataset_test = datasets.USPS('/data/usps', train=False, transform=None, 
                             target_transform=None,download=True)     
usps_test_image = osp.join(outdir, 'usps_test_image')
os.system("mkdir -p " + usps_test_image)
usps_labels = usps_dataset_test.targets #.labels.flatten()
with open(osp.join(outdir, 'usps_test.txt'), 'w') as label_file:
    for i in range(len(usps_test_image)):
        img = Image.fromarray(usps_dataset_test.data[i])
        img.save(osp.join(usps_test_image, '{:d}.png'.format(i)))
        label_file.write(usps_test_image+'/{:d}.png {:d}\n'.format(i, usps_labels[i]))