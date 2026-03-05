from .imagenet import get_imagenet_dataset
from .cifar10 import get_cifar10_dataset
from .cifar100 import get_cifar100_dataset
from .coco import get_coco_dataset

import copy
import torch
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    SequentialSampler,
    random_split
)
from mmdet.registry import DATA_SAMPLERS
from mmengine.registry import FUNCTIONS
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.dist import get_rank
from functools import partial
import numpy as np
from collections import defaultdict

def random_select(dataset, sample_num):
    n = len(dataset)
    subset_len = [sample_num, n - sample_num]
    subset, _ = random_split(
        dataset, subset_len, generator=torch.Generator()
    )
    return subset

def get_dataset(
        dataset_name, img_size = 224, 
        sub_label = None, task_type = "detection", 
        sample_num = -1
    ):
    if dataset_name == "imagenet":
        trainset, testset = get_imagenet_dataset(
            # root="/root/autodl-pub/datasets/imagenet",
            root = "/root/autodl-pvt/nuwa/data/dataset/imagenet",
            img_size=img_size,
            sub_label=sub_label
        )
    elif dataset_name == "cifar10":
        trainset, testset = get_cifar10_dataset(
            root="/root/autodl-pvt/nuwa/data/dataset/CIFAR10",
            img_size=img_size,
            sub_label=sub_label
        )
    elif dataset_name == "cifar100":
        trainset, testset = get_cifar100_dataset(
            root="/root/autodl-pvt/nuwa/data/dataset/CIFAR100",
            img_size=img_size,
            sub_label=sub_label
        )
    elif "coco" in dataset_name: # detection, segmentation
        trainset, testset = get_coco_dataset(
            root="/root/autodl-pub/datasets/coco",
            sub_label=sub_label,
            task_type=task_type
        )
    if sample_num>0:
        trainset = random_select(trainset, sample_num)
    
    return trainset, testset

def cfg2loader(dataset, cfg):
    dataloader_cfg = copy.deepcopy(cfg)
    # sampler
    sampler_cfg = dataloader_cfg.pop('sampler')
    sampler = DATA_SAMPLERS.build(
        sampler_cfg,
        default_args=dict(
            dataset=dataset, 
            seed=42
        )
    )
    # batch sampler
    batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler,
                batch_size=cfg.pop('batch_size')
            )
        )
    # collate_fn
    collate_fn = FUNCTIONS.get('pseudo_collate')
    # init_fn
    init_fn = partial(
        default_worker_init_fn,
        num_workers=dataloader_cfg.get('num_workers'),
        rank=get_rank(),
        seed=42,
        disable_subprocess_warning=False
    )
    # loader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
    )
    return dataloader

def get_dataloader(
        dataset_name, img_size = 224, 
        sub_label = None, task_type = "detection",
        train_batch_size = 64, eval_batch_size = 256,
        sample_num = -1,
    ):

    trainset, testset = get_dataset(dataset_name, img_size, sub_label, task_type, sample_num)
    if dataset_name != "coco":
        train_sampler = RandomSampler(trainset)
        test_sampler = SequentialSampler(testset)
        train_loader = DataLoader(
            trainset,sampler=train_sampler,batch_size=train_batch_size,
            num_workers=8,pin_memory=True,drop_last=False
        )
        test_loader = DataLoader(
            testset,sampler=test_sampler,batch_size=eval_batch_size,
            num_workers=8,pin_memory=True
        )
    else:
        train_dataloader_cfg = dict(
            batch_size=train_batch_size,
            num_workers=8,
            persistent_workers=True,
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_sampler=dict(type='AspectRatioBatchSampler'),
        )
        test_dataloader_cfg = dict(
            batch_size=eval_batch_size,
            num_workers=8,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_sampler=dict(type='AspectRatioBatchSampler'),
        )
        train_loader = cfg2loader(trainset, train_dataloader_cfg)
        test_loader = cfg2loader(testset, test_dataloader_cfg)
    return train_loader, test_loader