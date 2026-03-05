import random
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from time import perf_counter
import copy
import math

from engine.eval import evaluate
from dataset.utils import get_dataloader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_sub_task(args):
    path = os.path.join(args.output_dir,"sub_task",args.dataset_name,args.task_name+".txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    sub_task = [int(x.strip()) for x in content.split(",") if x.strip()]
    return sub_task

def is_identity_layer(attn: torch.nn.Module):
    has_params = any(p.numel() > 0 for p in attn.parameters())
    return not has_params

def get_gflops(model=None,config=None):
    if model is not None:
        if hasattr(model,"blocks"):
            N = model.pos_embed.shape[1]
            C = model.head.out_features
            p = model.patch_embed.patch_size[0]
            depth = len(model.blocks)
            for l in range(depth):
                if is_identity_layer(model.blocks[l]):
                    continue
                emb_dim = model.blocks[l].attn.qkv.in_features
                break

            flops = (3*(N-1)*p*p+2*depth*N+C+1)*emb_dim
            for l in range(depth):
                # attn
                if is_identity_layer(model.blocks[l]):
                    continue
                attn = model.blocks[l].attn
                if not is_identity_layer(attn):
                    num_heads = attn.num_heads
                    head_dim = attn.head_dim
                    qk_dim = attn.qk_dim if hasattr(attn,"qk_dim") else head_dim
                    v_dim = attn.vo_dim if hasattr(attn,"vo_dim") else head_dim
                    flops += (2*N*emb_dim+N*N)*(qk_dim+v_dim)*num_heads
                # mlp
                mlp = model.blocks[l].mlp
                mlp_dim = mlp.fc1.out_features
                flops += 2*N*emb_dim*mlp_dim
                
        elif hasattr(model,"backbone"):
            N_list = [60800,15200,3800,950]
            p = 4; C = model.roi_head.bbox_head.fc_cls.out_features
            flops = (3*(N_list[0]-1)*p*p+2*sum(N_list)+C+1)*768
            for s,stage in enumerate(model.backbone.stages):
                N = N_list[s]
                for block in stage.blocks:
                    emb_size = block.ffn.layers[0][0].in_features
                    num_heads = block.attn.w_msa.num_heads
                    head_dim = emb_size // num_heads
                    intermediate_size = block.ffn.layers[0][0].out_features
                    attn = block.attn.w_msa
                    qk_dim = attn.qk_dim if hasattr(attn,"qk_dim") else head_dim
                    v_dim = attn.vo_dim if hasattr(attn,"vo_dim") else head_dim
                    flops += (2*N*emb_size+N*N)*num_heads*(qk_dim+v_dim)
                    flops += 2*N*emb_size*intermediate_size

    elif config is not None:
        N = config["N"]
        C = config["C"]
        p = config["p"]
        depth = config["depth"]
        emb_dim = config["emb"]
        flops = (3*(N-1)*p*p+2*depth*N+C+1)*emb_dim
        for l in range(depth):
            # attn
            num_heads = config["head"][l]
            head_dim = config["head_dim"][l] if "head_dim" in config.keys() else 0
            qk_dim = config["qk"][l] if "qk" in config.keys() else head_dim
            v_dim = config["vo"][l] if "vo" in config.keys() else head_dim
            flops += (2*N*emb_dim+N*N)*(qk_dim+v_dim)*num_heads
            # mlp
            mlp_dim = config["mlp"][l]
            flops += 2*N*emb_dim*mlp_dim

    gflops = flops/1e9
    return gflops

def get_mparam(model):
    return sum(p.numel() for p in model.parameters())/1e6

@torch.no_grad()
def profiling(
        args,model,
        input_size=(256,3,224,224),
        warm_steps=3,run_steps=10,
    ):
    model.eval()
    print(f"Device: {args.device}")
    model.to(args.device)
    x = torch.randn(input_size).to(args.device)
    if next(model.parameters()).dtype == torch.half:
        x = x.half()
    print(f"Warming up model ({warm_steps} runs)...")
    for i in tqdm(range(warm_steps)):
        model(x)
        if args.device == "cuda":
            torch.cuda.synchronize()
    
    peak_memories = []  # GB
    times = []  # ms
    pbar = tqdm(range(run_steps))
    for i in pbar:
        torch.cuda.empty_cache()  
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = perf_counter()
        model(x)
        torch.cuda.synchronize()
        end_time = perf_counter()
        inference_time = (end_time - start_time)*1000
        times.append(inference_time)

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        peak_memories.append(peak_mem)

        pbar.set_postfix({
            "inference time (ms)": f"{inference_time:.2f}",
            "mem (GB)": f"{peak_mem:.2f}"
        })
    
    avg_inference_time = sum(times) / len(times)
    avg_peak_mem = sum(peak_memories) / len(peak_memories)

    inference_time = round(avg_inference_time,2)
    throughput = round(x.size(0) / (avg_inference_time) * 1000,2)
    peak_mem = round(avg_peak_mem,2)
    gflops = get_gflops(model)
    mparam = sum(p.numel() for p in model.parameters())/1e6

    profiling_dict = {
        "inference time (ms)": inference_time,
        "throughput (image/s)": throughput,
        "peak_memory (GB)": peak_mem,
        "#Params (M)": round(mparam,2),
        "FLOPs (G)": round(gflops,2),
    }

    return profiling_dict

def less_is_more(args,model,test_loader,repeat_num=1000):
    gflops = get_gflops(model)
    hidden_size = model.head.in_features
    N = model.pos_embed.shape[1]
    intermediate_size = model.blocks[0].mlp.fc1.out_features
    depth = len(model.blocks)
    results = []
    acc0 = evaluate(model,test_loader)
    cnt = 0
    for i in tqdm(range(repeat_num)):
        model_tmp = copy.deepcopy(model)
        pruning_rate = random.uniform(0.0, 0.2)
        prune_neuron_cnt = int(pruning_rate * gflops*1e9/(2*N*hidden_size))
        mask = torch.ones(intermediate_size * depth)
        indices = torch.randperm(mask.shape[0])[:prune_neuron_cnt]
        mask[indices] = 0
        mask = mask.view(depth,intermediate_size).to(args.device)
        for l in range(depth):
            model_tmp.blocks[l].mlp.fc2.weight.data *= mask[l]
        acc = evaluate(model_tmp,test_loader)
        results.append({
            "rate": pruning_rate,
            "acc": acc,
            "state": acc>acc0
        })
        if acc>acc0:
            cnt += 1
        print(f"Pruning Rate: {pruning_rate*100:.2f}, Acc: {acc:.4f}, State: {acc>acc0}")
        print(f"Positive Point: {cnt}")
    root = os.path.join(args.output_dir,"motivation")
    file_name = "less_is_more.pt"
    path = os.path.join(root,file_name)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    torch.save(results,path)

def get_score(args,model,metric="mag"):
    root = os.path.join(args.output_dir,"motivation")
    file_name = f"score_{metric}.pt"
    path = os.path.join(root,file_name)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    if os.path.exists(path):
        score_list = torch.load(path)
        return score_list
    
    train_loader, _ = get_dataloader(
        dataset_name = args.dataset_name,
        sub_label = args.sub_label,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        sample_num = args.sample_num,
    )
    depth = len(model.blocks)
    intermediate_size = model.blocks[0].mlp.fc1.out_features
    score_list = [torch.zeros(intermediate_size) for l in range(depth)]
    model.to(args.device)
    if metric=="mag":
        for l in range(depth):
            score_list[l] = model.blocks[l].mlp.fc1.weight.data.abs().mean(dim=1)+\
                model.blocks[l].mlp.fc2.weight.data.abs().mean(dim=0)
    elif metric=="act":
        def register_hook(layer_idx):
            def hook(module, input, output):
                score_list[layer_idx] += torch.abs((output.sum(dim=(0,1)))).cpu()
            return hook
        hooks = []
        for l in range(depth):
            hooks.append(model.blocks[l].mlp.act.register_forward_hook(register_hook(l)))
        with torch.no_grad():
            for batch in tqdm(train_loader):
                xb, yb = batch[:2]
                xb = xb.to(args.device)
                model(xb)
        for hook in hooks:
            hook.remove()
    elif metric=="grad":
        criterion = nn.CrossEntropyLoss()
        for batch in tqdm(train_loader):
            xb, yb = batch[:2]
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            out = model(xb)
            loss = criterion(out,yb)
            loss.backward()
        for l in range(depth):
            score_list[l] = model.blocks[l].mlp.fc1.weight.grad.data.abs().mean(dim=1)+\
                model.blocks[l].mlp.fc2.weight.grad.data.abs().mean(dim=0)
    elif metric=="taylor":
        criterion = nn.CrossEntropyLoss()
        for batch in tqdm(train_loader):
            xb, yb = batch[:2]
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            out = model(xb)
            loss = criterion(out,yb)
            loss.backward()
        for l in range(depth):
            W1 = model.blocks[l].mlp.fc1.weight.data
            G1 = model.blocks[l].mlp.fc1.weight.grad.data
            W2 = model.blocks[l].mlp.fc2.weight.data
            G2 = model.blocks[l].mlp.fc2.weight.grad.data
            score_list[l] = (W1*G1).abs().mean(dim=1)+(W2*G2).abs().mean(dim=0)
    torch.save(score_list,path)
    return score_list

def score_based_pruning(args,model,metric="mag"):
    gflops = get_gflops(model)
    N = model.pos_embed.shape[1]
    d = model.blocks[0].attn.qkv.in_features
    prune_neuron_count = int(gflops*1e9*args.pruning_rate/(2*N*d))
    args.train_batch_size = 128
    train_loader, _ = get_dataloader(
        dataset_name = args.dataset_name,
        sub_label = args.sub_label,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        sample_num = args.sample_num,
    )
    
    depth = len(model.blocks)
    score_list = get_score(args,model,metric)
    '''Global Pruning'''
    # for l in range(depth):
    #     score_list[l] = score_list[l].to(args.device)
    #     score_list[l] /= score_list[l].norm(p=2)
    # scores = torch.cat(score_list,dim=0)
    # threshold = torch.sort(scores)[0][prune_neuron_count]
    # for l in range(depth):
    #     mask = (score_list[l]>=threshold).float()
    #     mask = mask.to(model.blocks[l].mlp.fc2.weight.data.device)
    #     model.blocks[l].mlp.fc2.weight.data *= mask
    '''Uniform Pruning'''
    cnt = math.ceil(prune_neuron_count/depth)
    for l in range(depth):
        score_list[l] = score_list[l].to(args.device)
        indices = torch.topk(score_list[l],cnt,dim=0,largest=False).indices
        model.blocks[l].mlp.fc2.weight.data[:,indices] = 0
    return model

