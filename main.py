import warnings
warnings.filterwarnings("ignore")

import os

import torch
import argparse
import logging
import time

from utils.utils import *
from dataset.utils import *
from model.vit import *
from model.utils import *
from method.get_anchor_model import *
from engine.eval import *
from engine.train import *
from method.pruning import *
from method.nuwa import *

name2abb = {
    "deit_base_patch16_224": "deit_base",
    "deit_small_patch16_224": "deit_small",
    "deit_tiny_patch16_224": "deit_tiny",
    "mask_rcnn_swin_tiny": "swin_tiny",
    "vit_large_patch16_224": "vit_large",
    "bert_base_uncased": "bert_base",
}

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="NuWa", help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_name", type=str, default="deit_base_patch16_224", help="Model to use.")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset to use.")
    parser.add_argument("--task_name", type=str, default="T1-10", help="Sub-task of edge devices.")
    parser.add_argument("--task_type", type=str, default="recognition", help="Sub-task type of edge devices.")
    parser.add_argument("--pruning_rate", default=0.2, type=float,help="Pruning Rate.")
    parser.add_argument("--output_dir", default="/root/autodl-pvt/nuwa/data", type=str, help="The output directory where checkpoints will be written.")
    parser.add_argument("--baseline_dir", default="/root/autodl-pvt/nuwa/baseline/file", type=str, help="The directory to save the baseline results.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # Training Arguments
    parser.add_argument("--uniform", default=False, type=bool, help="Whether to use uniform pruning rates across all layers.")
    parser.add_argument("--sample_num", default=-1, type=int, help="Number of samples to use from the training set. -1 means using all samples.")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Total batch size for eval.")
    parser.add_argument("--num_epochs", default=1, type=int, help="Total number of training iteration to perform.")
    parser.add_argument("--num_steps", default=-1, type=int, help="Total number of training iteration to perform.")
    parser.add_argument("--eval_every", default=1000, type=int, help="Run prediction on validation set every so many steps.")
    parser.add_argument("--mask_lr", default=1e-3, type=float, help="The initial learning rate for mask params.")
    parser.add_argument("--beta_lr", default=1e-1, type=float, help="The initial learning rate for beta params.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=5e-2, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--accum_steps", default=1, type=float, help="Number of updates steps to accumulate before performing a backward/update pass.")

    # Pruning Arguments
    parser.add_argument("--anchor", default=True, type=bool, help="Whether to use anchor model for pruning.")
    parser.add_argument("--calibration_batch_size", default=128, type=int, help="Batch size for getting calibration features.")
    parser.add_argument("--mode", default="random", type=str, choices=["random","max"],help="Method to get calibration features.")
    parser.add_argument("--calibration_sample_num", default=128, type=int, help="Number of patches to use for calibration.")
    parser.add_argument("--energy_rate", default=0.93, type=float, help="Energy rate to preserve for SVD.")
    parser.add_argument("--target_arch", default="uniform", type=str, choices=["uniform","proportion", "adaptive"], help="Target Architecture.")
    parser.add_argument("--norm_type", default="l2", type=str, choices=["l2","min_max","z_score"], help="Normalization type for activation when target_arch=adaptive.")
    parser.add_argument("--strategy", default="optimize", type=str, choices=["prune","compensate","optimize"], help="Pruning Strategy for W2 in MLP modules.")

    return parser

def get_args():
    parser = get_args_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    args.local_rank = -1
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    args.name = f"{name2abb[args.model_name]}({args.dataset_name}-{args.task_name})"
    args.eval_epoch = True
    args.sub_label = get_sub_task(args)
    args.energy_rate = pruning2energy(args.pruning_rate)
    return args

def wandb_init(args):
    wandb.init(project='nuwa', name=args.name)
    config = wandb.config
    config.batch_size = args.train_batch_size
    config.test_batch_size = args.eval_batch_size
    config.epochs = args.num_epochs
    config.mask_lr = args.mask_lr
    config.beta_lr = args.beta_lr
    config.use_cuda = True  
    config.seed = args.seed  
    config.log_interval = 10
    wandb.watch_called = False 

def nuwa():
    args = get_args()
    model = get_model(
        args.model_name,args.dataset_name,
        root=os.path.join(args.output_dir,"param"),
        task_type=args.task_type,
    )
    _, test_loader = get_dataloader(
        dataset_name = args.dataset_name,
        sub_label = args.sub_label,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        sample_num = args.sample_num,
        task_type = args.task_type,
    )
    acc_original = evaluate(
        model,test_loader,
        visual_task=args.task_type,
        sub_label=args.sub_label
    )
    gflops_original = get_gflops(model)
    model_pruned=class_specific_derivation_nuwa(args,model,retrain=False)
    acc_pruned = evaluate(
        model_pruned,test_loader,
        visual_task=args.task_type,
        sub_label=args.sub_label
    )
    gflops_pruned = get_gflops(model_pruned)
    print(f"Original GFLOPS: {gflops_original:.2f}, New GFLOPS: {gflops_pruned:.2f}")
    print(f"Pruning Rate: {(gflops_original - gflops_pruned) / gflops_original:.2%}")
    print(f"Accuracy Original: {acc_original:.2f}, Accuracy Pruned: {acc_pruned:.2f}")
    
if __name__ == '__main__':
    nuwa()