import torch
import wandb
import os

name2abb = {
    "deit_base_patch16_224": "deit_base",
    "deit_small_patch16_224": "deit_small",
    "deit_tiny_patch16_224": "deit_tiny",
    "mask_rcnn_swin_tiny": "swin_tiny",
    "vit_large_patch16_224": "vit_large",
    "bert_base_uncased": "bert_base",
}

def wandb_log(step,**kwargs):
    if wandb.run is not None:
        wandb.log({
            **kwargs
        },step=step)

def wandb_log_acc(step, accuracy, pruned=False):
    if wandb.run is not None:
        if isinstance(accuracy,dict):
            wandb.log({
                f"{key}{'_pruned' if pruned else ''}": accuracy[key]
                for key in accuracy.keys()
            },step=step)
        else:
            wandb.log({
                f"accuracy{'_pruned' if pruned else ''}": accuracy
            },step=step)

def save_model(args,model,model_name=None):
    if model_name is None:
        model_name = f"{name2abb[args.model_name]}({args.task_name}-{args.pruning_rate:.1f}).pt"
    save_path = os.path.join(args.output_dir,"param",args.dataset_name,model_name)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    torch.save(model.state_dict(),save_path)

def load_model(args,model,model_name=None):
    if model_name is None:
        model_name = f"{name2abb[args.model_name]}({args.task_name}-{args.pruning_rate:.1f}).pt"
    load_path = os.path.join(args.output_dir,"param",args.dataset_name,model_name)
    model.load_state_dict(torch.load(load_path),strict=False)

def update_clusters(activation,clusters):
    '''
    clusters = [
        [
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            ...
        ],
        [
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            ...
        ],
        ...
    ]
    '''
    if isinstance(clusters[0][0], dict):
        for l in range(len(clusters)):
            if clusters[l] is None: continue
            for c in clusters[l]:
                if len(c["neurons"]) == 1: continue
                neuron_idxs = torch.tensor(c["neurons"])
                acts = activation[l][neuron_idxs]
                c["anchor_idx"] = neuron_idxs[acts.argmax().item()].item()
    elif isinstance(clusters[0][0], list):
        for s in range(len(clusters)):
            for b in range(len(clusters[s])):
                if clusters[s][b] is None: continue
                for c in clusters[s][b]:
                    if len(c["neurons"]) == 1: continue
                    neuron_idxs = torch.tensor(c["neurons"])
                    acts = activation[s][b][neuron_idxs]
                    c["anchor_idx"] = neuron_idxs[acts.argmax().item()].item()

def save_clusters(args,clusters):
    cluster_name = f"clusters({args.task_name}-{args.pruning_rate:.1f}-new).pt"
    save_path = os.path.join(args.output_dir,"cluster",args.dataset_name,args.model_name,cluster_name)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    torch.save(clusters,save_path)

def load_clusters(args):
    cluster_name = f"clusters({args.task_name}-{args.pruning_rate:.1f}-new).pt"
    load_path = os.path.join(args.output_dir,"cluster",args.dataset_name,args.model_name,cluster_name)
    clusters = torch.load(load_path)
    return clusters

def get_delta_acc(acc,acc_pruned):
    if isinstance(acc,dict):
        delta_acc = 0.0
        for key in acc.keys():
            delta_acc += abs(acc[key]-acc_pruned[key])
        delta_acc /= len(acc.keys())
    else:
        delta_acc = abs(acc-acc_pruned)
    return delta_acc

def better_acc(acc_pruned,best_acc):
    if isinstance(acc_pruned,dict):
        acc1 = 0.0
        for key in acc_pruned.keys():
            acc1 += acc_pruned[key]
        acc2 = 0.0
        for key in best_acc.keys():
            acc2 += best_acc[key]
        return acc1>acc2
    else:
        return acc_pruned>best_acc