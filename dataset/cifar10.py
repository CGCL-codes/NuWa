import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

def get_cifar10_dataset(
        root = "/share/hdd/CIFAR10", img_size = 224,
        sub_label = None,
    ):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=root,train=True,download=True,transform=transform_train)
    testset = datasets.CIFAR10(root=root,train=False,download=True,transform=transform_test)

    if sub_label != None:
        indices_train = [i for i, target in enumerate(trainset.targets) if target in sub_label]
        indices_test = [i for i, target in enumerate(testset.targets) if target in sub_label]
        trainset = Subset(trainset, indices_train)
        testset = Subset(testset, indices_test)

    return trainset, testset