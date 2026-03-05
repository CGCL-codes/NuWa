import torch
import os
from .vit import *
from .swin import *

def get_model(
        model_name = "deit_base_patch16_224",
        dataset_name = "imagenet",
        root = "/root/autodl-pvt/nuwa/data/param",
        task_type = "recongition",
    ):
    if "deit" in model_name:
        if model_name == "deit_base_patch16_224":
            if dataset_name == "imagenet":
                return deit_base()
            elif dataset_name == "cifar10":
                model = deit_base(num_classes=10)
                path = os.path.join(root,dataset_name,"deit_base(cifar10).pt")
                model.load_state_dict(torch.load(path))
                return model
            elif dataset_name == "cifar100":
                model = deit_base(num_classes=100)
                path = os.path.join(root,dataset_name,"deit_base(cifar100).pt")
                model.load_state_dict(torch.load(path))
                return model
        elif model_name == "deit_small_patch16_224":
            if dataset_name == "imagenet":
                return deit_small()
            elif dataset_name == "cifar10":
                model = deit_small(num_classes=10)
                path = os.path.join(root,dataset_name,"deit_small(cifar10).pt")
                model.load_state_dict(torch.load(path))
                return model
            elif dataset_name == "cifar100":
                model = deit_small(num_classes=100)
                path = os.path.join(root,dataset_name,"deit_small(cifar100).pt")
                model.load_state_dict(torch.load(path))
                return model
        elif model_name == "deit_tiny_patch16_224":
            if dataset_name == "imagenet":
                return deit_tiny()
            elif dataset_name == "cifar10":
                model = deit_tiny(num_classes=10)
                path = os.path.join(root,dataset_name,"deit_tiny(cifar10).pt")
                model.load_state_dict(torch.load(path))
                return model
            elif dataset_name == "cifar100":
                model = deit_tiny(num_classes=100)
                path = os.path.join(root,dataset_name,"deit_tiny(cifar100).pt")
                model.load_state_dict(torch.load(path))
                return model
    elif "swin" in model_name:
        if dataset_name == "coco":
            model = mask_rcnn_swin_t()
            if task_type == "detection":
                if hasattr(model.roi_head, 'mask_head'):
                    del model.roi_head.mask_head
                if hasattr(model.roi_head, 'mask_roi_extractor'):
                    del model.roi_head.mask_roi_extractor
                return model
            elif task_type == "segmentation":
                return model
            
    elif "vit" in model_name:
        if model_name == "vit_large_patch16_224":
            if dataset_name == "imagenet":
                return vit_large()
            elif dataset_name == "cifar10":
                model = vit_large(num_classes=10)
                path = os.path.join(root,dataset_name,"vit_large(cifar10).pt")
                model.load_state_dict(torch.load(path))
                return model
            elif dataset_name == "cifar100":
                model = vit_large(num_classes=100)
                path = os.path.join(root,dataset_name,"vit_large(cifar100).pt")
                model.load_state_dict(torch.load(path))
                return model
    
    elif "swin" in model_name:
        if dataset_name == "coco":
            model = mask_rcnn_swin_t()
            if task_type == "detection":
                if hasattr(model.roi_head, 'mask_head'):
                    del model.roi_head.mask_head
                if hasattr(model.roi_head, 'mask_roi_extractor'):
                    del model.roi_head.mask_roi_extractor
                return model
            elif task_type == "segmentation":
                return model
