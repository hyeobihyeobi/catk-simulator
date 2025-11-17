from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import Tensor
import math

config ={"num_experts_per_tok":6, 
        "n_routed_experts":64,
        "scoring_func":'softmax',
        "aux_loss_alpha":0.001,
        "seq_aux":False,
        "norm_topk_prob":True,
        "hidden_size": 128,
        "intermediate_size":128,
        "moe_intermediate_size": 80,
        "n_shared_experts":2,
        "deepseek_road_balance_loss": True,
        "Expert_select_by_AV_state_mode":None}

class MoEGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.top_k = config["num_experts_per_tok"]
        self.n_routed_experts = config["n_routed_experts"]

        self.scoring_func = config["scoring_func"]
        self.alpha = config["aux_loss_alpha"]
        self.seq_aux = config["seq_aux"]

        # topk selection algorithm
        self.norm_topk_prob = config["norm_topk_prob"]
        if config["Expert_select_by_AV_state_mode"]=="all_agent":
            self.gating_dim = config["hidden_size"]
        elif config["Expert_select_by_AV_state_mode"]=="AV_only":
            self.gating_dim = config["hidden_size"]
        elif config["Expert_select_by_AV_state_mode"]=="concat":
            self.gating_dim = int(config["hidden_size"]*2)
        else:
            raise NameError("unvalid Expert_selection_mode")
        self.linear = nn.Linear(self.n_routed_experts, self.gating_dim, bias=False)
        self.linear.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape # bsz->1, seq_len-> 
        ### compute gating score
        hidden_states = hidden_states.view(-1, h) # 5, 2048, batch*seq_len으로 
        logits = self.linear(hidden_states) # logits = 5, 64
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # 토큰마다 score 획득
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) # k=6, topk_weight.shape~5,6 / topk_idx.shape~5,6
        
        ### norm gate to sum 1 # 안함 # 근데 해볼수는 있을 듯 inference 때는 끄고, Train 때는 키는 구조인거 같은데
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0: # A100 40GB GPU?? 나중에..
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux: # 이걸로 그대로 구현해볼 수 있을듯
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else: # 이거 아님
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # print("이거한다고 되려나") # break는 안되도, print는 됨
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self):
        super().__init__()
        # self.config = config
        self.num_experts_per_tok = config["num_experts_per_tok"]
        self.experts = nn.ModuleList([DeepseekMLP(intermediate_size = config["moe_intermediate_size"]) for i in range(config["n_routed_experts"])])
        self.gate = MoEGate()
        if config["n_shared_experts"] is not None:
            intermediate_size = config["moe_intermediate_size"] * config["n_shared_experts"]
            self.shared_experts = DeepseekMLP(intermediate_size = intermediate_size)
    
    def forward(self, hidden_states):
        identity = hidden_states # 1, 24, 128
        orig_shape = hidden_states.shape
        
        if config["Expert_select_by_AV_state_mode"]=="all_agent":
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states) # topk_idx.shape ~ 5,6, topk_weight ~ 5,6
        elif config["Expert_select_by_AV_state_mode"]=="AV_only":
            av_hidden_states = hidden_states[:, :1, :].repeat(1, hidden_states.shape[-2], 1)
            topk_idx, topk_weight, aux_loss = self.gate(av_hidden_states)
        elif config["Expert_select_by_AV_state_mode"]=="concat":
            hybrid_hidden_states = torch.concat((hidden_states[:, :1, :].repeat(1, hidden_states.shape[-2], 1), \
                                                 hidden_states), dim=-1)
            topk_idx, topk_weight, aux_loss = self.gate(hybrid_hidden_states)
        else:
            raise NameError("unvalid Expert_selection_mode")
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) # 5, 2048
        flat_topk_idx = topk_idx.view(-1) # 30
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0) # self.num_experts_per_tok=6, 결과적으로 hidden_states
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts): # 여기서도 반복문이네
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            if config["deepseek_road_balance_loss"]:
                y = AddAuxiliaryLoss.apply(y, aux_loss) # 이게 되려나
            # y.retain_grad()
            self.y = y
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if config["n_shared_experts"] is not None:
            y = y + self.shared_experts(identity) # expert에 관계없이, 항상 적용되는 shared_expert 모델
        return y
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok # expert마다 부여되는 토큰 수 6개
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache

class DeepseekMLP(nn.Module):
    def __init__(self, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.hidden_size = config["hidden_size"] if hidden_size is None else hidden_size
        self.intermediate_size = config["intermediate_size"] if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        moeEncoder_num_experts_per_tok=6,
        moeEncoder_n_routed_experts=64,
        moeEncoder_scoring_func='softmax',
        moeEncoder_aux_loss_alpha=0.001,
        moeEncoder_seq_aux=False,
        moeEncoder_norm_topk_prob=True,
        moeEncoder_hidden_size=128,
        moeEncoder_intermediate_size=128,
        moeEncoder_n_shared_experts=2,
        moeEncoder_road_balance_loss=True,
        moe_intermediate_size=80,
        Expert_select_by_AV_state_mode=None,
    ):

        config["num_experts_per_tok"] = moeEncoder_num_experts_per_tok
        config["n_routed_experts"] = moeEncoder_n_routed_experts
        config["scoring_func"] = moeEncoder_scoring_func
        config["aux_loss_alpha"] = moeEncoder_aux_loss_alpha
        config["seq_aux"] = moeEncoder_seq_aux
        config["norm_topk_prob"] = moeEncoder_norm_topk_prob
        config["hidden_size"] = moeEncoder_hidden_size
        config["intermediate_size"] = moeEncoder_intermediate_size
        config["n_shared_experts"] = moeEncoder_n_shared_experts
        config["deepseek_road_balance_loss"] = moeEncoder_road_balance_loss
        config["moe_intermediate_size"] = moe_intermediate_size
        config["Expert_select_by_AV_state_mode"]=Expert_select_by_AV_state_mode

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        # self.mlp = DeepseekMoE()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights=False,
    ):
        src2 = self.norm1(src)
        src2, attn = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))

        if return_attn_weights:
            return src, attn

        return src