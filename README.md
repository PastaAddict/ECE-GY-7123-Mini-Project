# ECE-GY-7123-Mini_Project

* Kristi Topollai
* Beyza Kiper
* Baturalp Ozturk 

## Training residual architectures on the CIFAR10 dataset.


* Epochs : 500
* LR schedule: Cosine Annealing
* SGD-M: μ=0.9, LR=1e-2, nesterov=True, weight decay = 1e-4
* LAMB: LR=5e-3, weight decay = 0.02
* Augmentations
** A: No augmentation
** AB: with p=0.2 no augmentation with p=0.8 Mixup augmentation
** ABC: with p=0.2 no augmentation with p=0.8 -> with p=0.5 Mixup with p=0.5 Cutmix

### To run an experiment open train.py, and edit line 44, to change the parameters
 * Ex for resnet: arguments = ["--batch_size", "512" ,"--net_type", "resnet", "--num_blocks" , "4,3,3,0", "--optimizer", "lamb", "--augmentation", "ABC"]
 * Ex for pyramidnet: arguments = ["--batch_size", "2048" ,"--net_type", "pyramidnet", "--optimizer", "lamb", "--augmentation", "ABC"]
 After the training is over a .pth file will be created in the models directory with the name of the experiment (/home/kristi/models/--batch_size 512 --net_type resnet --num_blocks 2,2,2,0 --optimizer lamb --augmentation AB.pth). 

This file contains the parameters of the best performing model during the training process and the best accuracy on the test set.

### Extra required libraries
* torchsummary
* torch_optimizer

