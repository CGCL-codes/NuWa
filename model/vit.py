import torch
import timm

def deit_base(num_classes = 1000):
    model = timm.create_model('deit_base_patch16_224', pretrained=True)
    if num_classes != 1000:
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model

def deit_small(num_classes = 1000):
    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    if num_classes != 1000:
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model

def deit_tiny(num_classes = 1000):
    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    if num_classes != 1000:
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model

def vit_large(num_classes = 1000):
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    if num_classes != 1000:
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model