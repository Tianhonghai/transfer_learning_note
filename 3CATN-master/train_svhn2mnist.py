import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_list import ImageList
import os
from torch.autograd import Variable
import loss as loss_func
import numpy as np
import network
import net
import itertools
from utils import ReplayBuffer
import os.path as osp
import datetime

def train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, method,
          D_s, D_t, G_s2t, G_t2s, criterion_Sem, criterion_GAN, criterion_cycle, criterion_identity, optimizer_G,
          optimizer_D_t, optimizer_D_s,
          classifier1, classifier1_optim, fake_S_buffer, fake_T_buffer
          ):     
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        # data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        # data_target = data_target.cuda()

        optimizer.zero_grad()
        optimizer_ad.zero_grad()

        features_source,outputs_source =model(data_source)
        features_target,outputs_target = model(data_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        #feature, output = model(torch.cat((data_source, data_target), 0))

        loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, data_source.size(0)), label_source)
        softmax_output = nn.Softmax(dim=1)(outputs)

        output1 = classifier1(features)
        softmax_output1 = nn.Softmax(dim=1)(output1)
        softmax_output = (1-args.cla_plus_weight)*softmax_output+ args.cla_plus_weight*softmax_output1


        if epoch > start_epoch:
            if method == 'CDAN-E':
                entropy = loss_func.Entropy(softmax_output)
                loss += loss_func.CDAN([features, softmax_output], ad_net, entropy, network.calc_coeff(num_iter*(epoch-start_epoch)+batch_idx), random_layer)
            elif method == 'CDAN':
                loss += loss_func.CDAN([features, softmax_output], ad_net, None, None, random_layer)
            elif method == 'DANN':
                loss += loss_func.DANN(features, ad_net)
            else:
                raise ValueError('Method cannot be recognized.')

        # Cycle
        num_feature = features.size(0)
        # =================train discriminator T
        real_label = Variable(torch.ones(num_feature))
        # real_label = Variable(torch.ones(num_feature)).cuda()
        fake_label = Variable(torch.zeros(num_feature))
        # fake_label = Variable(torch.zeros(num_feature)).cuda()

        # 训练生成器
        optimizer_G.zero_grad()

        # Identity loss
        same_t = G_s2t(features_target)
        loss_identity_t = criterion_identity(same_t, features_target)

        same_s = G_t2s(features_source)
        loss_identity_s = criterion_identity(same_s, features_source)

        # Gan loss
        fake_t = G_s2t(features_source)
        pred_fake = D_t(fake_t)
        loss_G_s2t = criterion_GAN(pred_fake, label_source.float())

        fake_s = G_t2s(features_target)
        pred_fake = D_s(fake_s)
        loss_G_t2s = criterion_GAN(pred_fake, label_source.float())

        # cycle loss
        recovered_s = G_t2s(fake_t)
        loss_cycle_sts = criterion_cycle(recovered_s, features_source)

        recovered_t = G_s2t(fake_s)
        loss_cycle_tst = criterion_cycle(recovered_t, features_target)

        # sem loss
        pred_recovered_s = model.classifier(recovered_s)
        pred_fake_t = model.classifier(fake_t)
        loss_sem_t2s = criterion_Sem(pred_recovered_s, pred_fake_t)

        pred_recovered_t = model.classifier(recovered_t)
        pred_fake_s = model.classifier(fake_s)
        loss_sem_s2t = criterion_Sem(pred_recovered_t, pred_fake_s)

        loss_cycle = loss_cycle_tst + loss_cycle_sts
        weight_in_loss_g = args.weight_in_loss_g.split(',')
        loss_G = float(weight_in_loss_g[0]) * (loss_identity_s + loss_identity_t) + \
                 float(weight_in_loss_g[1]) * (loss_G_s2t + loss_G_t2s) + \
                 float(weight_in_loss_g[2])* loss_cycle + \
                 float(weight_in_loss_g[3]) * (loss_sem_s2t + loss_sem_t2s)


        # 训练softmax分类器
        outputs_fake = classifier1(fake_t.detach())
        # 分类器优化
        classifier_loss1 = nn.CrossEntropyLoss()(outputs_fake, label_source)
        classifier1_optim.zero_grad()
        classifier_loss1.backward()
        classifier1_optim.step()


        total_loss = loss + args.cyc_loss_weight * loss_G
        total_loss.backward()
        optimizer.step()
        optimizer_G.step()

        ###### Discriminator S ######
        optimizer_D_s.zero_grad()

        # Real loss
        pred_real = D_s(features_source.detach())
        loss_D_real = criterion_GAN(pred_real, real_label)

        # Fake loss
        fake_s = fake_S_buffer.push_and_pop(fake_s)
        pred_fake = D_s(fake_s.detach())
        loss_D_fake = criterion_GAN(pred_fake, fake_label)

        # Total loss
        loss_D_s = loss_D_real + loss_D_fake
        loss_D_s.backward()

        optimizer_D_s.step()
        ###################################

        ###### Discriminator t ######
        optimizer_D_t.zero_grad()

        # Real loss
        pred_real = D_t(features_target.detach())
        loss_D_real = criterion_GAN(pred_real, real_label)

        # Fake loss
        fake_t = fake_T_buffer.push_and_pop(fake_t)
        pred_fake = D_t(fake_t.detach())
        loss_D_fake = criterion_GAN(pred_fake, fake_label)

        # Total loss
        loss_D_t = loss_D_real + loss_D_fake
        loss_D_t.backward()
        optimizer_D_t.step()

        if epoch > start_epoch:
            optimizer_ad.step()
        if (batch_idx+epoch*num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss+G: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss.item(),total_loss.item()))

def test(args,epoch,config, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            # data, target = data.cuda(), target.cuda()
            feature, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    log_str = "epoch: {}, Accuracy: {}/{} ({:.4f}%)".format(
        epoch, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    config["out_file"].write(log_str + "\n")
    config["out_file"].flush()


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN SVHN MNIST')
    parser.add_argument('--method', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])
    parser.add_argument('--task', default='USPS2MNIST', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',  help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str, default='0', help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',  help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False, help='whether to use random')
    parser.add_argument('--output_dir',type=str,default="digits/s2m")
    parser.add_argument('--cla_plus_weight',type=float,default=0.3)
    parser.add_argument('--cyc_loss_weight',type=float,default=0.01)
    parser.add_argument('--weight_in_loss_g',type=str,default='1,0.01,0.1,0.1')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    source_list = '/data/svhn/svhn_train.txt'
    target_list = '/data/mnist/mnist_train.txt'
    test_list = '/data/mnist/mnist_test.txt'
    # train config
    config = {}

    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config['cyc_loss_weight'] = args.cyc_loss_weight
    config['cla_plus_weight'] = args.cla_plus_weight
    config['weight_in_loss_g'] = args.weight_in_loss_g
    config["epochs"] = args.epochs
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log_svhn_to_mnist_{}.txt".
                                       format(str(datetime.datetime.utcnow()))),
                              "w")

    config["out_file"].write(str(config))
    config["out_file"].flush()

    train_loader = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='RGB'),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)

    model = network.DTN()
    # model = model.cuda()
    class_num = 10

    #添加G，D，和额外的分类器
    z_dimension = 512
    D_s = network.models["Discriminator_digits"]()
    # D_s = D_s.cuda()
    G_s2t = network.models["Generator_digits"](z_dimension, 1024)
    # G_s2t = G_s2t.cuda()

    D_t = network.models["Discriminator_digits"]()
    # D_t = D_t.cuda()
    G_t2s = network.models["Generator_digits"](z_dimension, 1024)
    # G_t2s = G_t2s.cuda()

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_Sem = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(G_s2t.parameters(), G_t2s.parameters()), lr=0.0003)
    optimizer_D_s = torch.optim.Adam(D_s.parameters(), lr=0.0003)
    optimizer_D_t = torch.optim.Adam(D_t.parameters(), lr=0.0003)

    fake_S_buffer = ReplayBuffer()
    fake_T_buffer = ReplayBuffer()

    ## 添加分类器
    classifier1 = net.Net(512, class_num)
    # classifier1 = classifier1.cuda()
    classifier1_optim = optim.Adam(classifier1.parameters(), lr=0.0003)



    if args.random:
        random_layer = network.RandomLayer([model.output_num(), class_num], 500)
        ad_net = network.AdversarialNetwork(500, 500)
        # random_layer.cuda()
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(model.output_num() * class_num, 500)
    # ad_net = ad_net.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        if epoch % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.3
        train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, 0, args.method,
              D_s,D_t,G_s2t,G_t2s,criterion_Sem,criterion_GAN,criterion_cycle,criterion_identity,optimizer_G,optimizer_D_t,optimizer_D_s,
              classifier1,classifier1_optim,fake_S_buffer,fake_T_buffer)
        test(args,epoch,config, model, test_loader)

if __name__ == '__main__':
    main()
