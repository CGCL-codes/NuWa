import torch
import wandb
import torch.nn as nn
import os
import math

def save_mask(args,model_mask,pruning_rate):
    root = os.path.join(args.output_dir,"mask")
    if hasattr(model_mask, "uniform") and model_mask.uniform:
        file_name = f"{args.dataset_name}/{args.model_name}/mask({args.task_name}-uniform).pt"
    else:
        file_name = f"{args.dataset_name}/{args.model_name}/mask({args.task_name}).pt"
    path = os.path.join(root,file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(model_mask, "mha_masks"):
        mask_dict = {
            "mlp_masks": model_mask.mlp_masks,
            "mlp_betas": model_mask.mlp_betas,
            "mha_masks": model_mask.mha_masks,
            "mha_betas": model_mask.mha_betas,
            "rate": pruning_rate,
        }
    else:
        mask_dict = {
            "mlp_masks": model_mask.mlp_masks,
            "mlp_betas": model_mask.mlp_betas,
            "rate": pruning_rate,
        }

    torch.save(mask_dict,path)

def load_mask(args):
    root = os.path.join(args.output_dir,"mask")
    file_name = f"{args.dataset_name}/{args.model_name}/mask({args.task_name}{"-uniform" if args.uniform else ""}).pt"
    path = os.path.join(root,file_name)
    mask_dict = torch.load(path,weights_only=False)
    return mask_dict

def wandb_log(step,**kwargs):
    if wandb.run is not None:
        wandb.log({**kwargs},step=step)

def wandb_log_acc(step, accuracy):
    if wandb.run is not None:
        if isinstance(accuracy,dict):
            wandb.log({f"{key}": accuracy[key] for key in accuracy.keys()},step=step)
        else:
            wandb.log({f"accuracy": accuracy},step=step)

def better_acc(acc_1,acc_2):
    if isinstance(acc_1,dict):
        acc1 = 0.0
        for key in acc_1.keys():
            acc1 += acc_1[key]
        acc2 = 0.0
        for key in acc_2.keys():
            acc2 += acc_2[key]
        return acc1>=acc2
    else:
        return acc_1>=acc_2

def l2_norm(x):
    return x / x.norm(dim=-1, keepdim=True)

def min_max_norm(x):
    min = x.min()
    max = x.max()
    return (x-min)/(max-min)

def z_score_norm(x):
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x-mean)/std

def softmax_norm(x):
    T = x.abs().max().clamp(min=1e-6)
    return torch.softmax(x/T,dim=-1)

def normalization(x,norm_type="l2"):
    if norm_type == "l2":
        return l2_norm(x)
    elif norm_type == "min_max":
        return min_max_norm(x)
    elif norm_type == "z_score":
        return z_score_norm(x)
    elif norm_type == "softmax":
        return softmax_norm(x)
    else:
        raise ValueError(f"Unknown norm type {norm_type}")

def pruning2energy(pruning_rate):
    a = pruning_rate
    energy_rate = -0.41*math.pow(a,3)+0.14*math.pow(a,2)-0.03*a+1.0
    return energy_rate