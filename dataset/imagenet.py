import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Subset

class ImageNetFolder(datasets.ImageFolder):
    # used in get_imagenet to get the train_dataset or val_dataset
    def __init__(self, root, transform=None, layermap=None):
        super(ImageNetFolder, self).__init__(root, transform)
        self.layermap = layermap
    def __getitem__(self, index):
        original_tuple = super(ImageNetFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        image_id = os.path.splitext(os.path.basename(path))[0]
        xb=original_tuple[0]
        yb=original_tuple[1] if self.layermap==None else self.layermap[original_tuple[1]]
        return xb, yb, image_id

def get_imagenet_dataset(
        root = "/share/hdd/imagenet", img_size = 224,
        sub_label = None,
    ):
    transform_train=create_transform(
        input_size=img_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    transform_test = transforms.Compose([
        transforms.Resize(int(img_size / 0.875), interpolation=3),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    trainset = ImageNetFolder(root = f"{root}/train", transform = transform_train)
    testset = ImageNetFolder(root = f"{root}/val", transform = transform_test)
    if sub_label != None:
        indices_train = [i for i, target in enumerate(trainset.targets) if target in sub_label]
        indices_test = [i for i, target in enumerate(testset.targets) if target in sub_label]
        trainset = Subset(trainset, indices_train)
        testset = Subset(testset, indices_test)
    
    return trainset, testset
