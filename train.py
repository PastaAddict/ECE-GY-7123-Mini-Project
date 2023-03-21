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

import torchvision
import torchvision.transforms as transforms

from utils import progress_bar
from augmentations import mixup_data, cutmix_data, mix_criterion
from resnet import ResNet, BasicBlock
from pyramidnet import PyramidNet

import argparse

'''
Argparser for the different hyperparameters, note that not all hyperparams can be specified,
learning rate for example or the number of epochs are considered constant.
This was done in accordance to the way the experiments were conducted.
'''
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

parser.add_argument('--optimizer', default='sgd', type=str, choices = ['sgd', 'sgdm', 'adam', 'lamb'],
                    help='optimizer, SGD-M or LAMB')

parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')

parser.add_argument('--augmentation', default='AB', type=str, choices = ['A', 'AB', 'ABC'],
                    help='augmentation, no, no+mixup, no+mixup+cutmix')

print(parser)
arguments = ["--batch_size", "128" ,"--net_type", "resnet", "--num_blocks", "2,2,2,2",
             "--optimizer",  "sgdm", "--lr", "0.01", "--augmentation", "A"]
args = parser.parse_args(arguments)
print(args)


'''
Loading Cifar10 
'''
bsize = args.batch_size
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=int(bsize), shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


'''
Training (1 epoch) and testing functions
'''

def train(epoch, network, criterion, optimizer):

    global args
    if args.augmentation == 'A': # depending on the augmentation strategy, 
        p1, p2 = 1, 0            # the probabilities p1 and p2 are adjusted accordingly
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
          loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam) # the loss needs to be adjusted so that it accounts for mixed labels
          train_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0) # Accuracy on the training set
          correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                      + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
          
        else: # No mixing
          outputs = network(inputs)
          loss = criterion(outputs, targets)
          train_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0) # Accuracy on the training set
          correct += predicted.eq(targets).sum().item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return correct/total, train_loss/(batch_idx+1)

def test(epoch, network, criterion):
    global best_acc
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
        torch.save(state, './models/'+' '.join(arguments)+'.pth')
        best_acc = acc
    
    return correct/total, test_loss/(batch_idx+1)


# initialize the network
if args.net_type == 'resnet':
    net = ResNet(BasicBlock,[int(item) for item in args.num_blocks.split(',')]).to(device)
elif args.net_type == 'pyramidnet':
    net = PyramidNet('cifar10', args.depth, args.alpha, 10).to(device)

# Constant hyperparameters
criterion = nn.CrossEntropyLoss()
LR = args.lr
epochs = 200

#initialize the optimizer
if args.optimizer == 'sgdm':
    optimizer = optim.SGD(net.parameters(), lr=LR,
                        momentum=0.9)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=LR)

elif args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)
                        
elif args.optimizer == 'lamb':
    optimizer = optimizers.Lamb(net.parameters(), lr=LR, weight_decay = 0.02)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# main train loop

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
metrics = []
for epoch in range(start_epoch, start_epoch+epochs):
    train_acc, train_loss = train(epoch, net, criterion, optimizer)
    test_acc, test_loss = test(epoch, net, criterion)
    metrics.append([train_acc, train_loss, test_acc, test_loss])
    #scheduler.step()

state = torch.load('./models/'+' '.join(arguments)+'.pth')
state.update({'metrics': np.array(metrics)})
torch.save(state, './models/'+' '.join(arguments)+'.pth')