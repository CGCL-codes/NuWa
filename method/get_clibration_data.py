import torch
import os
from tqdm import tqdm
import pickle
import math

from dataset.utils import get_dataloader
from .utils import *

@torch.no_grad()
def get_activation_and_calibration_features(args,model_anchor):
    root1 = os.path.join(args.output_dir,"activation")
    file_name1 = f"{args.dataset_name}/{args.model_name}/activation({args.task_name}{"" if args.anchor else "original"}).pt"
    path1 = os.path.join(root1,file_name1)
    root2 = os.path.join(args.output_dir,"feature")
    sample_num = args.calibration_sample_num
    mode = args.mode
    file_name2 = f"{args.dataset_name}/{args.model_name}/feature({args.task_name}-{sample_num}-{mode}{"" if args.anchor else "original"}).pt"
    path2 = os.path.join(root2,file_name2)
    os.makedirs(os.path.dirname(path1), exist_ok=True)
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    if os.path.exists(path1) and os.path.exists(path2):
        activation = torch.load(path1)
        calibration_features = torch.load(path2,pickle_module=pickle)
        return activation, calibration_features
    '''prepare data'''
    train_loader, _ = get_dataloader(
        dataset_name = args.dataset_name,
        sub_label = args.sub_label,
        train_batch_size = args.calibration_batch_size,
        eval_batch_size = args.eval_batch_size,
        sample_num = args.sample_num,
        task_type = args.task_type,
    )
    hooks = []
    '''register hook function'''
    if hasattr(model_anchor, "blocks"):
        depth = model_anchor.depth
        activation = [torch.zeros(model_anchor.blocks[l].mlp.fc1.out_features,device=args.device) for l in range(depth)]
        calibration_features = [None for l in range(depth)]
        def register_hook(l):
            def hook_fn(module, input, output):
                activation[l] += torch.abs((output.sum(dim=(0,1))))
                features = output
                if mode == "random" and features.shape[0]>sample_num:
                    features = features[torch.randperm(features.shape[0])[:sample_num]]
                elif mode == "max" and features.shape[0]>sample_num:
                    l2_norm = (features ** 2).sum(dim=(1,2))
                    topk_indices = torch.topk(l2_norm,k=sample_num).indices
                    features = features[topk_indices]
                if calibration_features[l] is None:
                    calibration_features[l] = features
                else:
                    calibration_features[l] = torch.cat((calibration_features[l],features),dim=0)
                    if calibration_features[l].shape[0]>sample_num:
                        if mode == "random":
                            calibration_features[l] = calibration_features[l][torch.randperm(calibration_features[l].shape[0])[:sample_num]]
                        elif mode == "max":
                            l2_norm = (calibration_features[l] ** 2).sum(dim=(1,2))
                            topk_indices = torch.topk(l2_norm,k=sample_num).indices
                            calibration_features[l] = calibration_features[l][topk_indices]
            return hook_fn
        for l in range(depth):
            hooks.append(model_anchor.blocks[l].mlp.act.register_forward_hook(register_hook(l)))

    elif hasattr(model_anchor, "backbone"):
        activation = [[] for _ in range(len(model_anchor.backbone.stages))]
        calibration_features = [[] for _ in range(len(model_anchor.backbone.stages))]
        for s,stage in enumerate(model_anchor.backbone.stages):
            for block in stage.blocks:
                activation[s].append(torch.zeros(block.ffn.layers[0][0].out_features,device=args.device))
                calibration_features[s].append(None)
        scale_list = [64, 16, 4, 1]
        def register_hook(s,b):
            def hook_fn(module, inputs, output):
                token_num = args.calibration_sample_num*197
                activation[s][b] += torch.abs((output.sum(dim=(0,1))))
                features = output
                features = features[torch.randperm(features.shape[0])].reshape(-1,features.shape[-1])[:token_num]
                if calibration_features[s][b] is None:
                    calibration_features[s][b] = features
                else:
                    calibration_features[s][b] = torch.cat((calibration_features[s][b],features),dim=0)
                    if calibration_features[s][b].shape[0]>token_num:
                        calibration_features[s][b] = calibration_features[s][b][torch.randperm(calibration_features[s][b].shape[0])[:token_num]]
            return hook_fn
        for s,stage in enumerate(model_anchor.backbone.stages):
            for b,block in enumerate(stage.blocks):
                hooks.append(block.ffn.layers[0][1].register_forward_hook(register_hook(s,b)))

    '''get activation and calibration features'''
    model_anchor.to(args.device)
    for batch in tqdm(train_loader):
        if args.task_type == "recognition":
            xb, yb = batch[:2]
            xb = xb.to(args.device)
            model_anchor(xb)
        elif args.task_type == "detection":
            model_anchor.test_step(batch)
        elif args.task_type == "segmentation":
            model_anchor.test_step(batch)

    for h in hooks:
        h.remove()
    '''save activation and calibration features'''
    torch.save(activation, path1)
    torch.save(calibration_features, path2)
    return activation, calibration_features

def get_config_and_indices(args,model_anchor,prune_neuron_count,A):
    intermediate_size_list = []
    pruned_indices_list = []
    if args.target_arch == "adaptive":
        depth = model_anchor.depth
        for l in range(depth):
            A[l] = normalization(A[l],norm_type=args.norm_type)
        act_flat = torch.cat(A,dim=0)
        threshold = torch.kthvalue(act_flat,prune_neuron_count).values
        for l in range(depth):
            mask = A[l] >= threshold
            intermediate_size_list.append(int(mask.sum().item()))
            pruned_indices_list.append(torch.where(~mask)[0])

    elif args.target_arch == "uniform":
        if hasattr(model_anchor, "blocks"):
            depth = model_anchor.depth
            cnt_list = [A[l].shape[0] for l in range(depth)]
            total_cnt = sum(cnt_list)
            target_cnt = int((total_cnt-prune_neuron_count)/depth)
            intermediate_size_list = [0 for _ in range(depth)]
            sorted_indices = sorted(range(depth), key=lambda i: cnt_list[i])
            layer_count  = 0
            for i in sorted_indices:
                layer_count += 1
                if cnt_list[i]<target_cnt:
                    intermediate_size_list[i]=cnt_list[i]
                else:
                    intermediate_size_list[i]=target_cnt
                total_cnt -= cnt_list[i]
                prune_neuron_count -= (cnt_list[i]-intermediate_size_list[i])
                if depth > layer_count:
                    target_cnt = int((total_cnt - prune_neuron_count)/(depth - layer_count))
            '''==================='''
            for l in range(len(intermediate_size_list)):
                intermediate_size_list[l] = int(intermediate_size_list[l]//8*8)
            '''==================='''
            for l in range(depth):
                prune_cnt = cnt_list[l]-intermediate_size_list[l]
                indices = torch.topk(A[l], k=prune_cnt, largest=False).indices
                pruned_indices_list.append(indices)

            for l in range(depth):
                print(f"Layer {l}: MLP prune from {A[l].shape[0]} to {intermediate_size_list[l]}")

        elif hasattr(model_anchor, "backbone"):
            cnt_list = [[A[s][b].shape[0] for b in range(len(A[s]))] for s in range(len(A))]
            for s in range(len(cnt_list)):
                total_cnt = sum(cnt_list[s])
                prune_neuron_count_s = prune_neuron_count[s]
                target_cnt = int((total_cnt - prune_neuron_count_s)/len(cnt_list[s]))
                intermediate_size_list_s = [0 for _ in range(len(cnt_list[s]))]
                sorted_indices = sorted(range(len(cnt_list[s])), key=lambda i: cnt_list[s][i])
                layer_count  = 0
                for i in sorted_indices:
                    layer_count += 1
                    if cnt_list[s][i]<target_cnt:
                        intermediate_size_list_s[i]=cnt_list[s][i]
                    else:
                        intermediate_size_list_s[i]=target_cnt
                    total_cnt -= cnt_list[s][i]
                    prune_neuron_count_s -= (cnt_list[s][i]-intermediate_size_list_s[i])
                    if len(cnt_list[s]) > layer_count:
                        target_cnt = int((total_cnt - prune_neuron_count_s)/(len(cnt_list[s]) - layer_count))
                intermediate_size_list.append(intermediate_size_list_s)
            pruned_indices_list = [[] for _ in range(len(cnt_list))]
            for s in range(len(cnt_list)):
                for b in range(len(cnt_list[s])):
                    prune_cnt = cnt_list[s][b]-intermediate_size_list[s][b]
                    indices = torch.topk(A[s][b], k=prune_cnt, largest=False).indices
                    pruned_indices_list[s].append(indices)
            
            for s in range(len(cnt_list)):
                for b in range(len(cnt_list[s])):
                    print(f"Stage {s}, Block {b}: MLP prune from {A[s][b].shape[0]} to {intermediate_size_list[s][b]}")

    elif args.target_arch == "proportion":
        depth = model_anchor.depth
        cnt_list = [A[l].shape[0] for l in range(depth)]
        total_cnt = sum(cnt_list)
        for l in range(depth):
            intermediate_size = cnt_list[l]-int(cnt_list[l]/total_cnt*prune_neuron_count)
            intermediate_size_list.append(intermediate_size)
            prune_cnt = cnt_list[l]-intermediate_size_list[l]
            indices = torch.topk(A[l], k=prune_cnt, largest=False).indices
            pruned_indices_list.append(indices)

    
    return intermediate_size_list, pruned_indices_list