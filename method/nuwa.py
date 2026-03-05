import os

from utils.utils import get_gflops,get_sub_task
from dataset.utils import get_dataloader
from .get_anchor_model import get_anchor_model
from .pruning import pruning_vit
from engine.train import train

name2abb = {
    "deit_base_patch16_224": "deit_base",
    "deit_small_patch16_224": "deit_small",
    "deit_tiny_patch16_224": "deit_tiny",
    "mask_rcnn_swin_tiny": "swin_tiny",
}

def class_specific_derivation_nuwa(args,model,retrain=False):
    original_gflops = get_gflops(model)
    train_loader, test_loader = get_dataloader(
        dataset_name = args.dataset_name,
        sub_label = args.sub_label,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        sample_num = args.sample_num,
    )
    model_anchor = get_anchor_model(args,model,train_loader,test_loader)
    model_pruned = pruning_vit(args,model_anchor,original_gflops)
    if retrain:
        args.num_epochs = 10
        args.num_steps = -1
        args.train_batch_size = 128
        train_loader, test_loader = get_dataloader(
            dataset_name = args.dataset_name,
            sub_label = args.sub_label,
            train_batch_size = args.train_batch_size,
            eval_batch_size = args.eval_batch_size,
            sample_num = args.sample_num,
        )
        model_name = f"{name2abb[args.model_name]}({args.task_name}-{args.pruning_rate:.1f}-NuWa).pt"
        acc = train(args,model_pruned,train_loader,test_loader,model_name,early_stop=False,best=True)
        root = os.path.join(args.output_dir,"performance")
        os.makedirs(root, exist_ok=True)
        file_name = f"NuWa({args.task_name}-{args.pruning_rate:.1f}).txt"
        path = os.path.join(root,file_name)
        with open(path, "w") as f:
            f.write(f"{acc:.2f}")
    return model_pruned