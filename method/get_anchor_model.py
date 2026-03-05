import torch
import math
import torch.nn as nn
import logging
from tqdm import tqdm
from timm.layers.mlp import Mlp
import copy
import itertools

# Swin
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.backbones.swin import WindowMSA
# BERT
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.cache_utils import Cache

from .utils import *
from engine.eval import evaluate
from .pruning import AttentionPruned


class TopKBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs:torch.tensor, threshold):
        threshold = torch.sigmoid(threshold).item()
        mask = inputs.clone()
        _, idx = inputs.sort(descending=True)
        j = math.ceil(threshold * inputs.numel())
        mask[idx[j:]] = 0.
        mask[idx[:j]] = 1.
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output, ((grad_output * mask).sum()).view(-1)

class MaskLinear(nn.Linear):
    def __init__(self, weight, bias, mask, beta, mask_type="mlp", head_dim=64):
        super().__init__(weight.size(1), weight.size(0), bias=False)
        self.weight = weight
        self.bias = bias
        self.mask = mask
        self.beta = beta
        self.mask_type = mask_type
        self.head_dim = head_dim

    def forward(self, x):
        mask = TopKBinarizer.apply(self.mask, self.beta)
        if self.mask_type == "mha":
            mask = mask.repeat_interleave(self.head_dim)
        x = x * mask.view(1,1,-1)
        return nn.functional.linear(x, self.weight, self.bias)

class MaskViT(nn.Module):
    def __init__(self, model, uniform=False):
        super().__init__()
        self.model = model
        depth = len(self.model.blocks)
        self.intermediate_size = model.blocks[0].mlp.fc1.out_features
        self.num_heads = model.blocks[0].attn.num_heads
        for param in self.model.parameters():
            param.requires_grad = False
        self.uniform = uniform
        self.mlp_masks=nn.ParameterList([nn.Parameter(torch.ones(self.intermediate_size)) for _ in range(depth)])
        self.mha_masks=nn.ParameterList([nn.Parameter(torch.ones(self.num_heads)) for _ in range(depth)])
        if not uniform:
            self.mlp_betas=nn.ParameterList([nn.Parameter(torch.zeros(1)+5.0) for _ in range(depth)])
            self.mha_betas=nn.ParameterList([nn.Parameter(torch.zeros(1)+5.0) for _ in range(depth)])
        else:
            self.mlp_betas=nn.Parameter(torch.zeros(1)+5.0)
            self.mha_betas=nn.Parameter(torch.zeros(1)+5.0)
        self.replace_linear()

    def replace_linear(self):
        head_dim = self.model.blocks[0].attn.head_dim
        for l,block in enumerate(self.model.blocks):
            # MHA
            weight,bias = block.attn.proj.weight,block.attn.proj.bias
            mask = self.mha_masks[l]; beta = self.mha_betas[l] if not self.uniform else self.mha_betas
            block.attn.proj = MaskLinear(weight, bias, mask, beta, mask_type="mha", head_dim=head_dim)
            # MLP
            weight,bias = block.mlp.fc2.weight,block.mlp.fc2.bias
            mask = self.mlp_masks[l]; beta = self.mlp_betas[l] if not self.uniform else self.mlp_betas
            block.mlp.fc2 = MaskLinear(weight, bias, mask, beta, mask_type="mlp")
    
    def forward(self, x):
        return self.model(x)

class MaskViT_RCNN_Swin(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.mlp_masks=[
            [
                nn.Parameter(
                    torch.ones(block.ffn.layers[0][0].out_features)
                ) for block in stage.blocks
            ]
            for stage in model.backbone.stages
        ]
        self.mlp_betas=[
            [
                nn.Parameter(torch.zeros(1)+5.0) 
                for block in stage.blocks
            ] 
            for stage in model.backbone.stages
        ]
        self.replace_linear()
    
    def replace_linear(self):
        for s,stage in enumerate(self.model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                # MLP
                weight = block.ffn.layers[1].weight
                bias = block.ffn.layers[1].bias
                mask = self.mlp_masks[s][b]; beta = self.mlp_betas[s][b]
                block.ffn.layers[1] = MaskLinear(weight, bias, mask, beta, mask_type="mlp")
    
    def _run_forward(self, data, mode='loss'):
        return self.model._run_forward(data, mode)
    
    def test_step(self, batch):
        return self.model.test_step(batch) 

    def forward(self, x):
        data = self.model.data_preprocessor(x, True)
        losses = self.model._run_forward(data, mode='loss')
        return losses


def get_module_flops(model,model_name):
    if "deit" in model_name or "vit" in model_name:
        config = {
            "N": model.pos_embed.shape[1],
            "C": model.head.out_features,
            "p": model.patch_embed.patch_size[0],
            "depth": len(model.blocks),
            "emb": model.blocks[0].mlp.fc1.in_features,
            "head": model.blocks[0].attn.num_heads,
            "qk": model.blocks[0].attn.head_dim,
            "vo": model.blocks[0].attn.head_dim,
            "mlp": model.blocks[0].mlp.fc1.out_features,
        }
        N,p,C = config["N"],config["p"],config["C"]
        emb_head_gflops = (3*(N-1)*p*p+2*config["depth"]*N+C+1)*config["emb"]
        flops = emb_head_gflops
        mha_gflops = (2*N*config["emb"]+N*N)*config["head"]*(config["qk"]+config["vo"])
        mlp_gflops = 2*N*config["emb"]*config["mlp"]
        flops += (mha_gflops + mlp_gflops)*config["depth"]

        gflops = flops / 1e9
        mha_gflops = mha_gflops / 1e9
        mlp_gflops = mlp_gflops / 1e9
        emb_head_gflops = emb_head_gflops / 1e9

    elif "swin" in model_name:
        N_list = [60800,15200,3800,950]
        p = 4; C = model.roi_head.bbox_head.fc_cls.out_features
        emb_head_gflops = (3*(N_list[0]-1)*p*p+2*sum(N_list)+C+1)*768
        flops = emb_head_gflops
        mha_gflops = [[0 for _ in stage.blocks] for stage in model.backbone.stages]
        mlp_gflops = [[0 for _ in stage.blocks] for stage in model.backbone.stages]
        for s,stage in enumerate(model.backbone.stages):
            N = N_list[s]
            for b,block in enumerate(stage.blocks):
                emb_size = block.ffn.layers[0][0].in_features
                num_heads = block.attn.w_msa.num_heads
                head_dim = emb_size // num_heads
                intermediate_size = block.ffn.layers[0][0].out_features
                mha_gflops[s][b] += (2*N*emb_size+N*N)*num_heads*(head_dim+head_dim)
                mlp_gflops[s][b] += 2*N*emb_size*intermediate_size
                flops += mha_gflops[s][b] + mlp_gflops[s][b]

        gflops = flops / 1e9
        emb_head_gflops = emb_head_gflops / 1e9
        for s in range(len(model.backbone.stages)):
            for b in range(len(model.backbone.stages[s].blocks)):
                mha_gflops[s][b] = mha_gflops[s][b] / 1e9
                mlp_gflops[s][b] = mlp_gflops[s][b] / 1e9
        
    return gflops,mha_gflops,mlp_gflops,emb_head_gflops

def get_module_param(model,model_name):
    if "deit" in model_name or "vit" in model_name:
        pass
    elif "swin" in model_name:
        emb_head_mparam = sum([p.numel() for p in model.backbone.patch_embed.parameters()])
        mparam = emb_head_mparam
        mha_mparam = [[0 for _ in stage.blocks] for stage in model.backbone.stages]
        mlp_param = [[0 for _ in stage.blocks] for stage in model.backbone.stages]
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                mha_mparam[s][b] += sum([p.numel() for p in block.attn.parameters()])
                mlp_param[s][b] += sum([p.numel() for p in block.ffn.parameters()])
                mparam += mha_mparam[s][b] + mlp_param[s][b]
        mparam /= 1e6
        emb_head_mparam /= 1e6
        for s in range(len(model.backbone.stages)):
            for b in range(len(model.backbone.stages[s].blocks)):
                mha_mparam[s][b] /= 1e6
                mlp_param[s][b] /= 1e6
    return mparam,mha_mparam,mlp_param,emb_head_mparam

def get_overall_pruning_rate(model_mask,gflops,mha_gflops,mlp_gflops,emb_head_gflops):
    if isinstance(mlp_gflops, float):
        Rt = emb_head_gflops; depth = len(model_mask.model.blocks)
        for l in range(depth):
            mlp_beta = model_mask.mlp_betas[l] if not model_mask.uniform else model_mask.mlp_betas
            mha_beta = model_mask.mha_betas[l] if not model_mask.uniform else model_mask.mha_betas
            mlp_rate = math.ceil(torch.sigmoid(mlp_beta).item()*model_mask.intermediate_size)/model_mask.intermediate_size
            mha_rate = math.ceil(torch.sigmoid(mha_beta).item()*model_mask.num_heads)/model_mask.num_heads
            Rt += mlp_rate*mlp_gflops
            Rt += mha_rate*mha_gflops
        Rt/=gflops

    elif isinstance(mlp_gflops, list):
        Rt = emb_head_gflops
        for s,stage in enumerate(model_mask.model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                mlp_beta = model_mask.mlp_betas[s][b]
                intermediate_size = block.ffn.layers[0][0].out_features
                mlp_rate = math.ceil(torch.sigmoid(mlp_beta).item()*intermediate_size)/intermediate_size
                Rt += mlp_rate*mlp_gflops[s][b]
                Rt += mha_gflops[s][b]
        Rt/=gflops
    return (1-Rt)*100

def step_eval(
        args,global_setp,model_mask,
        test_loader,best_acc,best_rate,
        pruning_rate
    ):
    acc = evaluate(
        model_mask,test_loader,
        visual_task=args.task_type,
        sub_label=args.sub_label
    )
    wandb_log_acc(global_setp,acc)
    if better_acc(acc,best_acc):
        if best_acc == acc:
            best_rate = max(best_rate,pruning_rate)
        else:
            best_acc = acc
            best_rate = pruning_rate
    save_mask(args,model_mask,best_rate)
    return best_acc,best_rate
    
def train_mask(args,model,train_loader,test_loader,uniform=False):
    # train mask
    device = args.device
    if "deit" in args.model_name or "vit" in args.model_name:
        gflops,mha_gflops,mlp_gflops,emb_head_gflops = get_module_flops(model,args.model_name)
        model_mask = MaskViT(model,uniform)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model_mask.mlp_masks,  
                    "lr": args.mask_lr, 
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": model_mask.mlp_betas,  
                    "lr": args.beta_lr, 
                    "weight_decay": args.weight_decay,
                },
            ],
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
        )
    elif "swin" in args.model_name:
        gflops,mha_gflops,mlp_gflops,emb_head_gflops = get_module_param(model,args.model_name)
        model_mask = MaskViT_RCNN_Swin(model)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(itertools.chain(*model_mask.mlp_masks)), 
                    "lr": args.mask_lr, 
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": list(itertools.chain(*model_mask.mlp_betas)),
                    "lr": args.beta_lr, 
                    "weight_decay": args.weight_decay,
                },
            ],
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
        )
    model_mask.to(device)

    criterion = nn.CrossEntropyLoss()
    global_setp = 0
    if args.num_steps>0:
        num_steps = args.num_steps
    else:
        num_steps = args.num_epochs*len(train_loader)
    best_acc = evaluate(
        model_mask,test_loader,
        visual_task=args.task_type,
        sub_label=args.sub_label
    )
    best_rate = 0.0
    original_acc = best_acc
    wandb_log_acc(global_setp,best_acc)
    model_mask.zero_grad()

    print("\n=================================================================================")
    logger = logging.getLogger(__name__)
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    while True:
        model.train()
        epoch_iterator = tqdm(
            train_loader, 
            desc="Training (X / X Steps) (loss=X.X)", 
            bar_format="{l_bar}{r_bar}", dynamic_ncols=True
        )
        for batch in epoch_iterator:
            # forward+backward
            '''=========== Calculate the loss ============='''
            if args.task_type == "recognition":
                xb, yb = batch[:2]
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model_mask(xb)
                loss_task = criterion(logits, yb)
            elif args.task_type == "detection":
                data = model_mask.model.data_preprocessor(batch, True)
                losses = model_mask.model._run_forward(data, mode='loss')
                loss_task = sum(losses['loss_rpn_cls']) + sum(losses['loss_rpn_bbox']) + losses['loss_cls'] + losses['loss_bbox']
            elif args.task_type == "segmentation":
                data = model_mask.model.data_preprocessor(batch, True)
                losses = model_mask.model._run_forward(data, mode='loss')
                loss_task = sum(losses['loss_rpn_cls']) + sum(losses['loss_rpn_bbox']) + losses['loss_cls'] + losses['loss_bbox'] + losses['loss_mask']
            '''============================================'''
            loss = loss_task
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_setp += 1
            '''=========== Log ==========='''
            pruning_rate = get_overall_pruning_rate(
                model_mask,gflops,
                mha_gflops,mlp_gflops,
                emb_head_gflops
            )
            if isinstance(best_acc,dict):
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss_task=%2.4f) (Rate=%.2f) (bbox_mAP=%.1f) (segm_mAP=%.1f)" % (
                        global_setp, num_steps, loss_task.item(), pruning_rate, best_acc["bbox_mAP"],best_acc["segm_mAP"]
                    )
                )
            else:
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss_task=%2.4f) (Rate=%.2f) (best_acc=%.2f)" % (
                        global_setp, num_steps, loss_task.item(), pruning_rate, best_acc
                    )
                )
            wandb_log(
                global_setp,
                loss_task=loss_task.item(),
                pruning_rate=pruning_rate,
            )
            '''=========================='''
            if global_setp%args.eval_every==0:
                best_acc,best_rate = step_eval(
                    args,global_setp,model_mask,
                    test_loader,best_acc,best_rate,
                    pruning_rate
                )
            if global_setp>=num_steps:
                break
        if args.eval_epoch:
            best_acc,best_rate = step_eval(
                args,global_setp,model_mask,
                test_loader,best_acc,best_rate,
                pruning_rate
            )
        if global_setp>=num_steps:
            break
    
    if isinstance(best_acc,dict):
        logger.info("Original bbox_mAP: \t%f" % original_acc["bbox_mAP"])
        logger.info("Original segm_mAP: \t%f" % original_acc["segm_mAP"])
        logger.info("Best bbox_mAP: \t%f" % best_acc["bbox_mAP"])
        logger.info("Best segm_mAP: \t%f" % best_acc["segm_mAP"])
    else:
        logger.info("Original Accuracy: \t%f" % original_acc)
        logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("=================================================================================\n")
    return best_acc

def get_mask(args,model,train_loader,test_loader):
    uniform = args.uniform
    root = os.path.join(args.output_dir,"mask")
    file_name = f"{args.dataset_name}/{args.model_name}/mask({args.task_name}{"-uniform" if uniform else ""}).pt"
    path = os.path.join(root,file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        mask_dict = load_mask(args)
        return mask_dict
    train_mask(args,model,train_loader,test_loader,uniform)
    mask_dict = load_mask(args)
    return mask_dict

def mask2anchor(args,model,mask_dict):
    if hasattr(model, "blocks"):
        depth = len(model.blocks)
        W1_list = [model.blocks[l].mlp.fc1.weight.data for l in range(depth)]
        b1_list = [model.blocks[l].mlp.fc1.bias.data for l in range(depth)]
        W2_list = [model.blocks[l].mlp.fc2.weight.data for l in range(depth)]
        b2_list = [model.blocks[l].mlp.fc2.bias.data for l in range(depth)]
        WQKV_list = [model.blocks[l].attn.qkv.weight.data for l in range(depth)]
        bQKV_list = [model.blocks[l].attn.qkv.bias.data for l in range(depth)]
        WO_list = [model.blocks[l].attn.proj.weight.data for l in range(depth)]
        bO_list = [model.blocks[l].attn.proj.bias.data for l in range(depth)]
        for l in range(depth):
            # MHA
            mask = mask_dict["mha_masks"][l]
            beta = mask_dict["mha_betas"][l] if not args.uniform else mask_dict["mha_betas"]
            mask = TopKBinarizer.apply(mask, beta)
            indices = torch.nonzero(mask, as_tuple=False).squeeze().to(WQKV_list[l].device)
            head_dim = model.blocks[l].attn.head_dim
            num_heads = model.blocks[l].attn.num_heads
            if indices.numel()<num_heads:
                model.blocks[l].attn = AttentionPruned(
                    hidden_size = model.blocks[l].attn.qkv.in_features,
                    num_heads = indices.numel(),
                    qk_dim = head_dim, v_dim = head_dim,
                )
                print(f"Layer {l}: MHA prune {num_heads-indices.numel()} heads")
                attn_idx = torch.sort(torch.cat([
                    torch.arange(h * head_dim, (h + 1) * head_dim) for h in indices
                ])).values
                qkv_idx = []
                qkv_idx.append(attn_idx)
                qkv_idx.append(head_dim*num_heads+attn_idx)
                qkv_idx.append(2*head_dim*num_heads+attn_idx)
                qkv_idx = torch.sort(torch.cat(qkv_idx,dim=0)).values.to(WQKV_list[l].device)
                model.blocks[l].attn.qkv.weight.data.copy_(WQKV_list[l][qkv_idx])
                model.blocks[l].attn.qkv.bias.data.copy_(bQKV_list[l][qkv_idx])
                model.blocks[l].attn.proj.weight.data.copy_(WO_list[l][:,attn_idx])
                model.blocks[l].attn.proj.bias.data.copy_(bO_list[l])
            # MLP
            mask = mask_dict["mlp_masks"][l]
            beta = mask_dict["mlp_betas"][l] if not args.uniform else mask_dict["mlp_betas"]
            mask = TopKBinarizer.apply(mask, beta)
            indices = torch.nonzero(mask, as_tuple=False).squeeze().to(W1_list[l].device)
            if indices.numel()<W1_list[l].size(0):
                model.blocks[l].mlp = Mlp(
                    in_features=model.blocks[l].mlp.fc1.in_features,
                    hidden_features=indices.numel(),
                    out_features=model.blocks[l].mlp.fc2.out_features,
                )
                model.blocks[l].mlp.fc1.weight.data.copy_(W1_list[l][indices])
                model.blocks[l].mlp.fc1.bias.data.copy_(b1_list[l][indices])
                model.blocks[l].mlp.fc2.weight.data.copy_(W2_list[l][:,indices])
                model.blocks[l].mlp.fc2.bias.data.copy_(b2_list[l])
                print(f"Layer {l}: MLP prune {W1_list[l].size(0)-indices.numel()} neurons")
        model.hidden_size = model.blocks[0].mlp.fc1.in_features
        model.depth = len(model.blocks)
        model.patch_num = model.pos_embed.shape[1]
        
    elif hasattr(model, "backbone"):
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                W1 = copy.deepcopy(block.ffn.layers[0][0].weight.data)
                b1 = copy.deepcopy(block.ffn.layers[0][0].bias.data)
                W2 = copy.deepcopy(block.ffn.layers[1].weight.data)
                b2 = copy.deepcopy(block.ffn.layers[1].bias.data)
                embed_dims = block.ffn.layers[0][0].in_features
                mask = mask_dict["mlp_masks"][s][b]
                beta = mask_dict["mlp_betas"][s][b]
                mask = TopKBinarizer.apply(mask, beta)
                indices = torch.nonzero(mask, as_tuple=False).squeeze().to(W1.device)
                block.ffn = FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=indices.numel(),
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                )
                block.ffn.layers[0][0].weight.data.copy_(W1[indices])
                block.ffn.layers[0][0].bias.data.copy_(b1[indices])
                block.ffn.layers[1].weight.data.copy_(W2[:,indices])
                block.ffn.layers[1].bias.data.copy_(b2)
                print(f"Stage {s}, Block {b}: MLP prune {W1.size(0)-indices.numel()} neurons")
                
    return model

def head_focus(model,sub_label):
    if hasattr(model, "head"):
        cls_num = model.head.out_features
        all_idx = torch.arange(0, cls_num, dtype=torch.long)
        indices = all_idx[~torch.isin(all_idx, torch.tensor(sub_label))]
        model.head.weight.data[indices].zero_()
        model.head.bias.data[indices] = -1e9
    elif hasattr(model, "roi_head"):
        cls_num = model.roi_head.bbox_head.fc_cls.out_features
        all_idx = torch.arange(0, cls_num-1, dtype=torch.long) # background要保留
        indices = all_idx[~torch.isin(all_idx, torch.tensor(sub_label))]
        model.roi_head.bbox_head.fc_cls.weight.data[indices].zero_()
        model.roi_head.bbox_head.fc_cls.bias.data[indices] = -1e9
    return model

def get_anchor_model(args,model,train_loader,test_loader,focus=True):
    print("================== Get Anchor Model ==================")
    mask_dict = get_mask(args,model,train_loader,test_loader)
    print(f"Pruning Rate: {mask_dict['rate']:.2f}%\n")
    model_anchor = mask2anchor(args,model,mask_dict)
    if focus:
        model_anchor = head_focus(model_anchor,args.sub_label)
    return model_anchor






