import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable
import torch_optimizer as optimizers


import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_number = torch.cuda.device_count()

import torchvision
import torchvision.transforms as transforms

from utils import progress_bar
from augmentations import mixup_data, cutmix_data, mix_criterion
from resnet import ResNet, BasicBlock
from pyramidnet import PyramidNet

import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--net_type', default='pyramidnet', type=str, choices = ['resnet', 'pyramidnet'],
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('--depth', default=110, type=int,
                    help='depth of the network (default: 110)')
parser.add_argument('--alpha', default=84, type=int,
                    help='alpha of pyramid network (default: 84)')
parser.add_argument('--num_blocks', default='1,1,1,1', type=str,
                    help='resnet architecture')

parser.add_argument('--optimizer', default='sgd', type=str, choices = ['sgd', 'lamb'],
                    help='optimizer, SGD-M or LAMB')

parser.add_argument('--augmentation', default='AB', type=str, choices = ['A', 'AB', 'ABC'],
                    help='augmentation, no, no+mixup, no+mixup+cutmix')

print(parser)
arguments = ["--batch_size", "256" ,"--net_type", "resnet", 
             "--num_blocks" , "4,3,3,0", "--optimizer",  
             "sgd", "--augmentation", "ABC"]
args = parser.parse_args(arguments)


# Data
bsize = args.batch_size
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=int(bsize/gpu_number), shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# all together

def train(epoch, network, criterion, optimizer):

    global args
    if args.augmentation == 'A':
        p1, p2 = 1, 0
    elif args.augmentation == 'AB':
        p1, p2 = 0.2, 1
    elif args.augmentation == 'ABC':
        p1, p2 = 0.2, 0.5

    

    print('\nEpoch: %d' % epoch)
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if np.random.random()>p1:
          if np.random.random()>p2:

            ''' With CutMix'''
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets,
                                                          1.0, True)
          else:

            ''' With MixUp'''
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           0.1, True)
          inputs, targets_a, targets_b = map(Variable, (inputs,
                                                        targets_a, targets_b))
          outputs = network(inputs)
          loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
          train_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                      + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
          
        else:
          outputs = network(inputs)
          loss = criterion(outputs, targets)
          train_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, network, criterion, optimizer):
    global best_acc
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': network.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        current_directory = os.getcwd()
        print(current_directory)
        final_directory = os.path.join(current_directory, r'models')
        if not os.path.isdir(final_directory):
            os.mkdir(final_directory)
        #modelname = ' '.join(arguments)
        torch.save(state, './models/'+' '.join(arguments)+'.pth')
        best_acc = acc



#net = PyramidNet('cifar10', 110, 84, 10).to(device)
if args.net_type == 'resnet':
    net = ResNet(BasicBlock,[int(item) for item in args.num_blocks.split(',')]).to(device)
    net = nn.DataParallel(net)
elif args.net_type == 'pyramidnet':
    net = PyramidNet('cifar10', args.depth, args.alpha, 10).to(device)


criterion = nn.CrossEntropyLoss()
LR = 1e-2
epochs = 500

if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=LR,
                        momentum=0.9, weight_decay=1e-4)
elif args.optimizer == 'lamb':
    optimizer = optimizers.Lamb(net.parameters(), lr=LR, weight_decay = 0.02)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
#print(net, optimizer)
for epoch in range(start_epoch, start_epoch+epochs):
    train(epoch, net, criterion, optimizer)
  
    test(epoch, net, criterion, optimizer)
    scheduler.step()