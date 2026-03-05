import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
import os
import math
import copy

# DeiT
from timm.layers.mlp import Mlp
from timm.layers.attention import Attention,maybe_add_mask
# Swin
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.backbones.swin import WindowMSA

from .get_clibration_data import *
from utils.utils import get_gflops,get_mparam

class AttentionPruned(Attention):
    def __init__(self, hidden_size, num_heads, qk_dim, vo_dim, qkv_bias=True):
        super().__init__((hidden_size//num_heads)*num_heads, num_heads, qkv_bias)
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.vo_dim = vo_dim
        self.scale = self.qk_dim ** -0.5
        self.qkv = nn.Linear(hidden_size,  num_heads*(qk_dim*2+vo_dim), bias=qkv_bias)
        self.proj = nn.Linear(vo_dim*num_heads, hidden_size)

    def forward(self,x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x) # [B,N,num_heads*(qk_dim*2+vo_dim)]
        q = qkv[:,:,:self.qk_dim*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        k = qkv[:,:,self.qk_dim*self.num_heads:self.qk_dim*2*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        v = qkv[:,:,self.qk_dim*2*self.num_heads:].reshape(B,N,self.num_heads,self.vo_dim).permute(0,2,1,3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.vo_dim*self.num_heads)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowMSAPruned(WindowMSA):
    def __init__(self, embed_dims, num_heads, window_size, qk_dim, v_dim, qkv_bias=True):
        super().__init__(
            embed_dims=embed_dims, 
            num_heads=num_heads, 
            window_size=window_size, 
            qkv_bias=qkv_bias
        )
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.scale = self.qk_dim ** -0.5
        self.qkv = nn.Linear(embed_dims,  num_heads*(qk_dim*2+v_dim), bias=qkv_bias)
        self.proj = nn.Linear(v_dim*num_heads, embed_dims)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        qkv = self.qkv(x) # [B,N,num_heads*(qk_dim*2+v_dim)]
        q = qkv[:,:,:self.qk_dim*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        k = qkv[:,:,self.qk_dim*self.num_heads:self.qk_dim*2*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        v = qkv[:,:,self.qk_dim*2*self.num_heads:].reshape(B,N,self.num_heads,self.v_dim).permute(0,2,1,3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.v_dim*self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def get_compensation(X ,W ,pruned_indices):
    device = W.device
    D = W.shape[0]
    k = len(pruned_indices)

    Mp = torch.zeros((D, k), device=device)
    for i, idx in enumerate(pruned_indices):
        Mp[idx, i] = 1.0

    # G = (2 X^T X + γ I)^(-1)
    gamma = 1e-3
    XTX = X.T @ X
    I = torch.eye(D, device=device)
    G = torch.linalg.solve(2 * XTX + gamma*I, I)

    M_middle = Mp.T @ G @ Mp  # shape: (k, k)
    I = torch.eye(k, device=device)
    M_middle_inv = torch.linalg.solve(M_middle + gamma*I, I)

    delta_W = - G @ Mp @ M_middle_inv @ Mp.T @ W
    return delta_W

def prune_W2(args,X,W2,pruned_indices):
    mask = torch.ones(W2.shape[1], dtype=torch.bool)
    mask[pruned_indices] = False
    retrain_indices = torch.where(mask)[0]
    if args.strategy=="prune":
        W2_pruned = W2[:,retrain_indices]
    elif args.strategy=="compensate":
        delta_W = get_compensation(X, W2.T, pruned_indices)
        W2 += delta_W.T
        W2_pruned = W2[:,retrain_indices]
    elif args.strategy=="optimize":
        X = X.to(args.device)
        W2 = W2.to(args.device)
        retrain_indices = retrain_indices.to(args.device)
        Y = X@W2.T # [N,d]
        X_prime = X[:,retrain_indices] # [N,e']
        X_prime_pinv = torch.linalg.pinv(X_prime) # [e',N]
        W2_pruned = Y.T @ X_prime_pinv.T

    return W2_pruned

def prune_mlp(args,model_anchor,prune_neuron_count):
    A,Xs = get_activation_and_calibration_features(args,model_anchor)
    # get mlp pruning config
    intermediate_size_list,pruned_indices_list = get_config_and_indices(args,model_anchor,prune_neuron_count,A)
    if hasattr(model_anchor,"blocks"):
        depth = model_anchor.depth
        W1_list = [model_anchor.blocks[l].mlp.fc1.weight.data for l in range(depth)]
        b1_list = [model_anchor.blocks[l].mlp.fc1.bias.data for l in range(depth)]
        W2_list = [model_anchor.blocks[l].mlp.fc2.weight.data for l in range(depth)]
        b2_list = [model_anchor.blocks[l].mlp.fc2.bias.data for l in range(depth)]
        for l in tqdm(range(depth)):
            pruned_indices = pruned_indices_list[l]
            X = Xs[l].to(W2_list[l].device).view(-1,Xs[l].shape[-1])
            '''============='''
            # U, S, Vh = svd(X)
            # Vh = Vh[:intermediate_size_list[l], :]
            # scores = (Vh.T ** 2).sum(dim=1)
            # pruned_indices = torch.argsort(scores, descending=False)[:len(pruned_indices)]
            '''============='''
            W2_pruned = prune_W2(args,X,W2_list[l],pruned_indices)
            model_anchor.blocks[l].mlp = Mlp(
                in_features=model_anchor.blocks[l].mlp.fc1.in_features,
                hidden_features=intermediate_size_list[l],
                out_features=model_anchor.blocks[l].mlp.fc2.out_features,
            )
            mask = torch.ones(W1_list[l].shape[0], dtype=torch.bool)
            mask[pruned_indices] = False
            retrain_indices = torch.where(mask)[0]
            model_anchor.blocks[l].mlp.fc1.weight.data.copy_(W1_list[l][retrain_indices])
            model_anchor.blocks[l].mlp.fc1.bias.data.copy_(b1_list[l][retrain_indices])
            model_anchor.blocks[l].mlp.fc2.weight.data.copy_(W2_pruned)
            model_anchor.blocks[l].mlp.fc2.bias.data.copy_(b2_list[l])

    elif hasattr(model_anchor,"backbone"):
        for s,stage in enumerate(model_anchor.backbone.stages):
            for b,block in enumerate(stage.blocks):
                embed_dims = block.ffn.layers[0][0].in_features
                W1 = copy.deepcopy(block.ffn.layers[0][0].weight.data)
                b1 = copy.deepcopy(block.ffn.layers[0][0].bias.data)
                W2 = copy.deepcopy(block.ffn.layers[1].weight.data)
                b2 = copy.deepcopy(block.ffn.layers[1].bias.data)
                pruned_indices = pruned_indices_list[s][b]
                X = Xs[s][b].to(W2.device).view(-1,Xs[s][b].shape[-1])
                W2_pruned = prune_W2(args,X,W2,pruned_indices)
                block.ffn = FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=intermediate_size_list[s][b],
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                )
                mask = torch.ones(W1.shape[0], dtype=torch.bool)
                mask[pruned_indices] = False
                retrain_indices = torch.where(mask)[0]
                block.ffn.layers[0][0].weight.data.copy_(W1[retrain_indices])
                block.ffn.layers[0][0].bias.data.copy_(b1[retrain_indices])
                block.ffn.layers[1].weight.data.copy_(W2_pruned)
                block.ffn.layers[1].bias.data.copy_(b2)


def svd(W):
    try:
        U, Sigma, Vh = torch.linalg.svd(W)
    except RuntimeError:
        print("SVD failed, fallback to gesvd with regularization")
        eps = 1e-6 * torch.eye(W.shape[0], device=W.device)
        U, Sigma, Vh = torch.linalg.svd(W + eps, driver="gesvd")
    return U,Sigma,Vh

def prune_mha(args,model_anchor):
    energy_rate = args.energy_rate
    qk_dim_list, vo_dim_list = [],[]
    root_dim = os.path.join(args.output_dir,"config")
    file_name_qk = f"{args.dataset_name}/{args.model_name}/qk_dim_list({energy_rate:.2f}).pt"
    file_name_vo = f"{args.dataset_name}/{args.model_name}/vo_dim_list({energy_rate:.2f}).pt"
    path_qk_dim = os.path.join(root_dim,file_name_qk)
    path_vo_dim = os.path.join(root_dim,file_name_vo)
    os.makedirs(os.path.dirname(path_qk_dim), exist_ok=True)
    os.makedirs(os.path.dirname(path_vo_dim), exist_ok=True)
    if os.path.exists(path_qk_dim) and os.path.exists(path_vo_dim):
        qk_dim_list = torch.load(path_qk_dim)
        vo_dim_list = torch.load(path_vo_dim)

    if hasattr(model_anchor,"blocks"):
        depth = model_anchor.depth
        hidden_size = model_anchor.hidden_size
        head_dim = model_anchor.blocks[0].attn.head_dim
        WQKV_list = [model_anchor.blocks[l].attn.qkv.weight.data for l in range(depth)]
        bQKV_list = [model_anchor.blocks[l].attn.qkv.bias.data for l in range(depth)]
        WO_list = [model_anchor.blocks[l].attn.proj.weight.data for l in range(depth)]
        bO_list = [model_anchor.blocks[l].attn.proj.bias.data for l in range(depth)]
        for l in range(depth):
            block = model_anchor.blocks[l]
            num_heads = block.attn.num_heads
            WQ = WQKV_list[l][:hidden_size].reshape(num_heads,head_dim,hidden_size)
            bQ = bQKV_list[l][:hidden_size].reshape(num_heads,head_dim)
            WK = WQKV_list[l][hidden_size:hidden_size*2].reshape(num_heads,head_dim,hidden_size)
            bK = bQKV_list[l][hidden_size:hidden_size*2].reshape(num_heads,head_dim)
            WV = WQKV_list[l][hidden_size*2:].reshape(num_heads,head_dim,hidden_size)
            bV = bQKV_list[l][hidden_size*2:].reshape(num_heads,head_dim)
            WO = WO_list[l].T.reshape(num_heads,head_dim,hidden_size)
            root = os.path.join(args.output_dir,"svd")
            file_name_qk = f"{args.dataset_name}/{args.model_name}/qk_results({l}).pt"
            file_name_vo = f"{args.dataset_name}/{args.model_name}/vo_results({l}).pt"
            path_qk = os.path.join(root,file_name_qk)
            path_vo = os.path.join(root,file_name_vo)
            os.makedirs(os.path.dirname(path_qk), exist_ok=True)
            os.makedirs(os.path.dirname(path_vo), exist_ok=True)
            if os.path.exists(path_qk) and os.path.exists(path_vo):
                qk_results = torch.load(path_qk)
                vo_results = torch.load(path_vo)
                U_qk_list, S_qk_list, Vh_qk_list = qk_results["U"],qk_results["S"],qk_results["Vh"]
                U_vo_list, S_vo_list, Vh_vo_list = vo_results["U"],vo_results["S"],vo_results["Vh"]
            else:
                print(f"Calculating SVD Results for layer {l}...")
                U_qk_list, S_qk_list, Vh_qk_list = [],[],[]
                U_vo_list, S_vo_list, Vh_vo_list = [],[],[]
                for h in tqdm(range(num_heads)):
                    WQ_hat = torch.cat((WQ[h],bQ[h].unsqueeze(1)),dim=1)
                    WK_hat = torch.cat((WK[h],bK[h].unsqueeze(1)),dim=1)
                    WV_hat = torch.cat((WV[h],bV[h].unsqueeze(1)),dim=1)
                    U, Sigma, Vh = svd(WQ_hat.T @ WK_hat)
                    U_qk_list.append(U);S_qk_list.append(Sigma);Vh_qk_list.append(Vh)
                    U, Sigma, Vh = svd(WV_hat.T @ WO[h])
                    U_vo_list.append(U);S_vo_list.append(Sigma);Vh_vo_list.append(Vh)
                qk_results = {"U":U_qk_list,"S":S_qk_list,"Vh":Vh_qk_list}
                vo_results = {"U":U_vo_list,"S":S_vo_list,"Vh":Vh_vo_list}
                torch.save(qk_results, path_qk)
                torch.save(vo_results, path_vo)
            
            # get qk_dim and vo_dim
            if len(qk_dim_list)==depth and len(vo_dim_list)==depth:
                qk_dim, vo_dim = qk_dim_list[l], vo_dim_list[l]
            else:
                print(f"Getting qk_dim and vo_dim for layer {l}...")
                qk_rate,vo_rate = 1.0, 1.0
                qk_dim,vo_dim = head_dim, head_dim
                qk_energy_list = [torch.sum(S_qk_list[h]**2).item() for h in range(num_heads)]
                vo_energy_list = [torch.sum(S_vo_list[h]**2).item() for h in range(num_heads)]
                while qk_rate >= energy_rate:
                    qk_dim -= 1
                    retain_energy_list = [torch.sum(S_qk_list[h][:qk_dim]**2).item() for h in range(num_heads)]
                    qk_rate = sum([retain_energy_list[h]/qk_energy_list[h] for h in range(num_heads)])/num_heads
                qk_dim += 1;qk_dim_list.append(qk_dim)
                while vo_rate >= energy_rate:
                    vo_dim -= 1
                    retain_energy_list = [torch.sum(S_vo_list[h][:vo_dim]**2).item() for h in range(num_heads)]
                    vo_rate = sum([retain_energy_list[h]/vo_energy_list[h] for h in range(num_heads)])/num_heads
                vo_dim += 1;vo_dim_list.append(vo_dim)
            '''==================='''
            qk_dim = int(qk_dim//8*8)
            vo_dim = int(vo_dim//8*8)
            '''==================='''
            # prune mha
            print(f"Layer {l}: MHA prune from {head_dim} to {qk_dim} for QK, from {head_dim} to {vo_dim} for VO")
            WQ_new_list,bQ_new_list = [],[]
            WK_new_list,bK_new_list = [],[]
            WV_new_list,bV_new_list = [],[]
            WO_new_list = []
            for h in range(num_heads):
                U, Sigma, Vh = U_qk_list[h].to(args.device), S_qk_list[h].to(args.device), Vh_qk_list[h].to(args.device)
                WQ_hat_new = (U[:,:qk_dim] @ torch.sqrt(torch.diag(Sigma[:qk_dim]))).T * math.sqrt(qk_dim/head_dim)
                WK_hat_new = torch.sqrt(torch.diag(Sigma[:qk_dim])) @ Vh[:qk_dim,:]
                WQ_new_list.append(WQ_hat_new[:,:-1])
                bQ_new_list.append(WQ_hat_new[:,-1].T)
                WK_new_list.append(WK_hat_new[:,:-1])
                bK_new_list.append(WK_hat_new[:,-1].T)
                U, Sigma, Vh = U_vo_list[h].to(args.device), S_vo_list[h].to(args.device), Vh_vo_list[h].to(args.device)
                WV_hat_new = (U[:,:vo_dim] @ torch.sqrt(torch.diag(Sigma[:vo_dim]))).T
                WO_hat_new = torch.sqrt(torch.diag(Sigma[:vo_dim])) @ Vh[:vo_dim,:]
                WV_new_list.append(WV_hat_new[:,:-1])
                bV_new_list.append(WV_hat_new[:,-1].T)
                WO_new_list.append(WO_hat_new)
            model_anchor.blocks[l].attn = AttentionPruned(
                hidden_size,
                num_heads=num_heads,
                qk_dim=qk_dim,
                vo_dim=vo_dim,
                qkv_bias=True,
            )
            WQKV_new = torch.cat((torch.cat(WQ_new_list,dim=0),torch.cat(WK_new_list,dim=0),torch.cat(WV_new_list,dim=0)),dim=0)
            bQKV_new = torch.cat((torch.cat(bQ_new_list,dim=0),torch.cat(bK_new_list,dim=0),torch.cat(bV_new_list,dim=0)),dim=0)
            WO_new = torch.cat(WO_new_list,dim=0).T
            model_anchor.blocks[l].attn.qkv.weight.data.copy_(WQKV_new)
            model_anchor.blocks[l].attn.qkv.bias.data.copy_(bQKV_new)
            model_anchor.blocks[l].attn.proj.weight.data.copy_(WO_new)
            model_anchor.blocks[l].attn.proj.bias.data.copy_(bO_list[l])
        # save qk_dim_list and vo_dim_list
        if not os.path.exists(path_qk_dim) or not os.path.exists(path_vo_dim):
            torch.save(qk_dim_list, path_qk_dim)
            torch.save(vo_dim_list, path_vo_dim)

    elif hasattr(model_anchor,"backbone"):
        if not os.path.exists(path_qk_dim) or not os.path.exists(path_vo_dim):
            qk_dim_list = [[] for stage in model_anchor.backbone.stages]
            vo_dim_list = [[] for stage in model_anchor.backbone.stages]
        for s,stage in enumerate(model_anchor.backbone.stages):
            for b,block in enumerate(stage.blocks):
                embed_dims = block.attn.w_msa.embed_dims
                num_heads = block.attn.w_msa.num_heads
                window_size = block.attn.w_msa.window_size
                head_dim = embed_dims // num_heads
                WQKV = copy.deepcopy(block.attn.w_msa.qkv.weight.data)
                bQKV = copy.deepcopy(block.attn.w_msa.qkv.bias.data)
                WO = copy.deepcopy(block.attn.w_msa.proj.weight.data)
                bO = copy.deepcopy(block.attn.w_msa.proj.bias.data)
                WQ = WQKV[:embed_dims].reshape(num_heads,head_dim,embed_dims)
                bQ = bQKV[:embed_dims].reshape(num_heads,head_dim)
                WK = WQKV[embed_dims:embed_dims*2].reshape(num_heads,head_dim,embed_dims)
                bK = bQKV[embed_dims:embed_dims*2].reshape(num_heads,head_dim)
                WV = WQKV[embed_dims*2:].reshape(num_heads,head_dim,embed_dims)
                bV = bQKV[embed_dims*2:].reshape(num_heads,head_dim)
                WO = WO.T.reshape(num_heads,head_dim,embed_dims)
                relative_position_bias_table = copy.deepcopy(block.attn.w_msa.relative_position_bias_table.data)

                root = os.path.join(args.output_dir,"svd")
                file_name_qk = f"{args.dataset_name}/{args.model_name}/qk_results({s}-{b}).pt"
                file_name_vo = f"{args.dataset_name}/{args.model_name}/vo_results({s}-{b}).pt"
                path_qk = os.path.join(root,file_name_qk)
                path_vo = os.path.join(root,file_name_vo)
                os.makedirs(os.path.dirname(path_qk), exist_ok=True)
                os.makedirs(os.path.dirname(path_vo), exist_ok=True)
                if os.path.exists(path_qk) and os.path.exists(path_vo):
                    qk_results = torch.load(path_qk)
                    vo_results = torch.load(path_vo)
                    U_qk_list, S_qk_list, Vh_qk_list = qk_results["U"],qk_results["S"],qk_results["Vh"]
                    U_vo_list, S_vo_list, Vh_vo_list = vo_results["U"],vo_results["S"],vo_results["Vh"]
                else:
                    print(f"Calculating SVD Results for stage {s} block {b}...")
                    U_qk_list, S_qk_list, Vh_qk_list = [],[],[]
                    U_vo_list, S_vo_list, Vh_vo_list = [],[],[]
                    for h in tqdm(range(num_heads)):
                        WQ_hat = torch.cat((WQ[h],bQ[h].unsqueeze(1)),dim=1)
                        WK_hat = torch.cat((WK[h],bK[h].unsqueeze(1)),dim=1)
                        WV_hat = torch.cat((WV[h],bV[h].unsqueeze(1)),dim=1)
                        U, Sigma, Vh = svd(WQ_hat.T @ WK_hat)
                        U_qk_list.append(U);S_qk_list.append(Sigma);Vh_qk_list.append(Vh)
                        U, Sigma, Vh = svd(WV_hat.T @ WO[h])
                        U_vo_list.append(U);S_vo_list.append(Sigma);Vh_vo_list.append(Vh)
                    qk_results = {"U":U_qk_list,"S":S_qk_list,"Vh":Vh_qk_list}
                    vo_results = {"U":U_vo_list,"S":S_vo_list,"Vh":Vh_vo_list}
                    torch.save(qk_results, path_qk)
                    torch.save(vo_results, path_vo)
                
                if os.path.exists(path_qk_dim) and os.path.exists(path_vo_dim):
                    qk_dim, vo_dim = qk_dim_list[s][b], vo_dim_list[s][b]
                else:
                    print(f"Getting qk_dim and vo_dim for stage {s} block {b}...")
                    qk_rate,vo_rate = 1.0, 1.0
                    qk_dim,vo_dim = head_dim, head_dim
                    qk_energy_list = [torch.sum(S_qk_list[h]**2).item() for h in range(num_heads)]
                    vo_energy_list = [torch.sum(S_vo_list[h]**2).item() for h in range(num_heads)]
                    while qk_rate >= energy_rate:
                        qk_dim -= 1
                        retain_energy_list = [torch.sum(S_qk_list[h][:qk_dim]**2).item() for h in range(num_heads)]
                        qk_rate = sum([retain_energy_list[h]/qk_energy_list[h] for h in range(num_heads)])/num_heads
                    qk_dim += 1;qk_dim_list[s].append(qk_dim)
                    while vo_rate >= energy_rate:
                        vo_dim -= 1
                        retain_energy_list = [torch.sum(S_vo_list[h][:vo_dim]**2).item() for h in range(num_heads)]
                        vo_rate = sum([retain_energy_list[h]/vo_energy_list[h] for h in range(num_heads)])/num_heads
                    vo_dim += 1;vo_dim_list[s].append(vo_dim)
                    print(f"pruning qk_dim from {head_dim} to: {qk_dim}, vo_dim from {head_dim} to: {vo_dim}")
                
                WQ_new_list,bQ_new_list = [],[]
                WK_new_list,bK_new_list = [],[]
                WV_new_list,bV_new_list = [],[]
                WO_new_list = []
                for h in range(num_heads):
                    WQ_hat = torch.cat((WQ[h],bQ[h].unsqueeze(1)),dim=1)
                    WK_hat = torch.cat((WK[h],bK[h].unsqueeze(1)),dim=1)
                    WV_hat = torch.cat((WV[h],bV[h].unsqueeze(1)),dim=1)
                    # pruning qk_size
                    U, Sigma, Vh = svd(WQ_hat.T @ WK_hat)
                    WQ_hat_new = (U[:,:qk_dim] @ torch.sqrt(torch.diag(Sigma[:qk_dim]))).T * math.sqrt(qk_dim/head_dim)
                    WK_hat_new = torch.sqrt(torch.diag(Sigma[:qk_dim])) @ Vh[:qk_dim,:]
                    WQ_new_list.append(WQ_hat_new[:,:-1])
                    bQ_new_list.append(WQ_hat_new[:,-1].T)
                    WK_new_list.append(WK_hat_new[:,:-1])
                    bK_new_list.append(WK_hat_new[:,-1].T)
                    # pruning vo_size
                    U, Sigma, Vh = svd(WV_hat.T @ WO[h])
                    WV_hat_new = (U[:,:vo_dim] @ torch.sqrt(torch.diag(Sigma[:vo_dim]))).T
                    WO_hat_new = torch.sqrt(torch.diag(Sigma[:vo_dim])) @ Vh[:vo_dim,:]
                    WV_new_list.append(WV_hat_new[:,:-1])
                    bV_new_list.append(WV_hat_new[:,-1].T)
                    WO_new_list.append(WO_hat_new)
                WQKV_new = torch.cat((torch.cat(WQ_new_list,dim=0),torch.cat(WK_new_list,dim=0),torch.cat(WV_new_list,dim=0)),dim=0)
                bQKV_new = torch.cat((torch.cat(bQ_new_list,dim=0),torch.cat(bK_new_list,dim=0),torch.cat(bV_new_list,dim=0)),dim=0)
                WO_new = torch.cat(WO_new_list,dim=0).T

                block.attn.w_msa = WindowMSAPruned(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    window_size=window_size,
                    qk_dim=qk_dim,
                    v_dim=vo_dim,
                    qkv_bias=True,
                )
                block.attn.w_msa.qkv.weight.data.copy_(WQKV_new)
                block.attn.w_msa.qkv.bias.data.copy_(bQKV_new)
                block.attn.w_msa.proj.weight.data.copy_(WO_new)
                block.attn.w_msa.proj.bias.data.copy_(bO)
                block.attn.w_msa.relative_position_bias_table.data.copy_(relative_position_bias_table)

        # save qk_dim_list and vo_dim_list
        if not os.path.exists(path_qk_dim) or not os.path.exists(path_vo_dim):
            torch.save(qk_dim_list, path_qk_dim)
            torch.save(vo_dim_list, path_vo_dim)

def pruning_vit(args,model_anchor,original_size,mha=True,mlp=True):
    pruning_rate = args.pruning_rate
    target_size = original_size * (1 - pruning_rate)
    model_anchor.to(args.device)
    print("================== Pruning MHA ==================")
    if mha:
        prune_mha(args,model_anchor)

    print("================== Pruning MLP ==================")
    if hasattr(model_anchor,"blocks"):
        current_size = get_gflops(model_anchor)
        print(f"Pruning rate after MHA pruning: {1-current_size/original_size:.2%}")
        prune_neuron_count = int((current_size - target_size)*1e9/(2*model_anchor.patch_num*model_anchor.hidden_size))
        if prune_neuron_count>0 and mlp:
            prune_mlp(args,model_anchor,prune_neuron_count)

    elif hasattr(model_anchor,"backbone"):
        current_size = get_mparam(model_anchor.backbone)
        print(f"Pruning rate after MHA pruning: {1-current_size/original_size:.2%}")
        prune_neuron_count = [0 for _ in range(len(model_anchor.backbone.stages))]
        left_mparam = (current_size - target_size)*1e6
        param_rate = [0 for _ in range(len(model_anchor.backbone.stages))]

        for s,stage in enumerate(model_anchor.backbone.stages):
            for block in stage.blocks:
                param_rate[s] += sum([p.numel() for p in block.parameters()])
        for s,stage in enumerate(model_anchor.backbone.stages):
            embed_dim = stage.blocks[0].attn.w_msa.embed_dims
            param = left_mparam*(param_rate[s]/sum(param_rate))
            prune_neuron_count[s] = int(param/(2*embed_dim))

        if sum(prune_neuron_count)>0 and mlp:
            prune_mlp(args,model_anchor,prune_neuron_count)
    
    return model_anchor
            

            


        
        
        

        


