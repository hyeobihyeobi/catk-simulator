from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding
from ..layers.mlp_layer import MLPLayer
import math
import torch.nn.functional as F
from .agent_predictor import AgentPredictor, DistPredictor

config ={"num_experts_per_tok":6, 
         "n_routed_experts":64,
         "scoring_func":'softmax',
         "aux_loss_alpha":0.001,
         "seq_aux":True,
         "norm_topk_prob":True,
         "hidden_size": 128,
         "intermediate_size":128,
         "moe_intermediate_size":80,
         "n_shared_experts":2,
         "first_k_dense_replace":1,
         "deepseek_road_balance_loss": True,
         'ifreturn_logit': False,}

class DecoderLayerRouter(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout, Distance_attn = False, history_steps=21, reduce_ffn_dim=False,) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
            
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        if reduce_ffn_dim:
            self.ffn = nn.Sequential(
                nn.Linear(dim, config["moe_intermediate_size"]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(config["moe_intermediate_size"], dim),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim * mlp_ratio, dim),
            )
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.Distance_attn = Distance_attn
        if self.Distance_attn:
            self.gen_tau = nn.Linear(dim, num_heads)
        self.history_steps = history_steps


    def forward(
        self,
        tgt,
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
        data = None,
        history_steps=21,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape

        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt2 = self.norm1(tgt)
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask.repeat(M, 1)
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)
        tgt_valid = tgt_tmp[tgt_valid_mask]
        tgt2_valid = self.norm2(tgt_valid)
        tgt2_valid, _ = self.m2m_attn(
            tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid
        )
        tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
        tgt = torch.zeros_like(tgt_tmp)
        tgt[tgt_valid_mask] = tgt_valid

        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        
        if self.Distance_attn:
            position = torch.cat([data['agent']['position'][:, :, self.history_steps - 1], data["map"]["polygon_center"][..., :2], data["static_objects"]["position"]], dim=1)
            dist = torch.norm(position[:,0:1] - position, dim=-1)
            tau = self.gen_tau(tgt2).permute(0,2,1)
            attn_mask = dist[:, None,:,None] * tau[:,:,None]  # [B, 8, Q, Q]
            attn_mask = attn_mask.permute(0,2,1,3)
            attn_mask[memory_key_padding_mask] = float("-inf")
            attn_mask = attn_mask.permute(0,2,3,1).flatten(0,1)
            tgt2 = self.cross_attn(tgt2, memory, memory, attn_mask=attn_mask)[0]
        else:
            tgt2 = self.cross_attn(tgt2, memory, memory, key_padding_mask=memory_key_padding_mask)[0]
            
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)

        return tgt

class MoEGate(nn.Module):
    def __init__(self,
                 router_init_weight = "False",
                 router_init_bias = "False",
                 router_original_init = False,
                 ):
        super().__init__()
        self.config = config
        self.top_k = config["num_experts_per_tok"]
        self.n_routed_experts = config["n_routed_experts"]

        self.scoring_func = config["scoring_func"]
        self.alpha = config["aux_loss_alpha"]
        self.seq_aux = config["seq_aux"]
        self.router_init_weight = router_init_weight
        self.router_init_bias = router_init_bias
        # topk selection algorithm
        self.norm_topk_prob = config["norm_topk_prob"]
        self.gating_dim = config["hidden_size"]
        if router_original_init:
            self.init_weight_yet = True
        else:
            self.init_weight_yet = False
        self.linear = nn.Linear(self.n_routed_experts, self.gating_dim, bias=False)
        self.linear.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        if router_init_weight != "False":
            self.init_weight_yet = True
            if router_init_bias != "False":
                self.linear = nn.Linear(self.gating_dim, self.n_routed_experts, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape # bsz->1, seq_len-> 
        ### compute gating score
        hidden_states = hidden_states.view(-1, h) # 5, 2048, batch*seq_len으로 
        
        if self.init_weight_yet:
            if self.router_init_weight != "False":
                self.linear.weight.data.fill_(self.router_init_weight)
                if self.router_init_bias != "False":
                    self.linear.bias.data.fill_(0.1)
            else:
                import torch.nn.init  as init
                init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
            self.init_weight_yet = False

        logits = self.linear(hidden_states) # logits = 5, 64

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) # k=6, topk_weight.shape~5,6 / topk_idx.shape~5,6
        
        ### norm gate to sum 1 
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0: 
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        if config['ifreturn_logit']:
            return topk_idx, topk_weight, aux_loss, logits
        else:
            return topk_idx, topk_weight, aux_loss, scores

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
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device) * config["deepseek_road_balance_loss_weight"]
        return grad_output, grad_loss
    
class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, dim, num_heads, mlp_ratio, dropout, layer_idx, history_steps, deepseek_residual,
                 router_init_weight = "False",
                 router_init_bias = "False",
                 router_original_init = False,
                 shared_expert_weight = 1.0,
                 deepseek_n_shared_experts = None,
                 ):
        super().__init__()
        self.deepseek_residual = deepseek_residual
        self.shared_expert_weight = shared_expert_weight
        self.deepseek_n_shared_experts = deepseek_n_shared_experts
        
        self.num_experts_per_tok = config["num_experts_per_tok"]
        self.experts = nn.ModuleList([DeepseekMLP(intermediate_size = config["moe_intermediate_size"]) for i in range(config["n_routed_experts"])])
        
        self.gate = MoEGate(router_init_weight, router_init_bias, router_original_init)            
        if deepseek_n_shared_experts is not None:
            intermediate_size = config["moe_intermediate_size"] * config["n_shared_experts"]
            self.shared_experts = DeepseekMLP(intermediate_size = intermediate_size)
        
    def forward(self, hidden_states, tgt_route=None):
        identity = hidden_states # 1, 24, 128
        orig_shape = hidden_states.shape            
        topk_idx, topk_weight, aux_loss, scores = self.gate(tgt_route) # topk_idx.shape ~ 5,6, topk_weight ~ 5,6
        hidden_states_ = hidden_states.view(-1, hidden_states.shape[-1]) # 5, 2048
        flat_topk_idx = topk_idx.view(-1) # 30
        if self.training:
            hidden_states_ = hidden_states_.repeat_interleave(self.num_experts_per_tok, dim=0) # self.num_experts_per_tok=6, 결과적으로 hidden_states
            y = torch.empty_like(hidden_states_)
            for i, expert in enumerate(self.experts): # 여기서도 반복문이네
                y[flat_topk_idx == i] = expert(hidden_states_[flat_topk_idx == i]) # expert마다 할당된 토큰들에 대해서만 expert 연산 수행
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            if config["deepseek_road_balance_loss"]:
                y = AddAuxiliaryLoss.apply(y, aux_loss) # 이게 되려나
            # y.retain_grad()
            self.y = y
        else:
            y = self.moe_infer(hidden_states_, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.deepseek_n_shared_experts is not None:
            y = y + (self.shared_expert_weight * self.shared_experts(identity)) # expert에 관계없이, 항상 적용되는 shared_expert 모델
            
        if self.deepseek_residual:
            y = y + hidden_states
        return y, scores
    
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

    def forward(self, x, tgt_route=None):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout, layer_idx, history_steps = 21, future_steps = 80, \
                    deepseek_residual = False, use_av_query_for_displacement = False,\
                            router_init_weight = "False",
                            router_init_bias = "False",
                            router_original_init = False,
                            shared_expert_weight = 1.0,
                            deepseek_n_shared_experts = False,
                            ) -> None:
        super().__init__()
        self.dim = dim
        self.deepseek_residual = deepseek_residual
        self.use_av_query_for_displacement = use_av_query_for_displacement
        self.shared_expert_weight = shared_expert_weight
        self.deepseek_n_shared_experts = deepseek_n_shared_experts
        
        
        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.router = DecoderLayerRouter(dim, num_heads, mlp_ratio, dropout,)

        if (layer_idx >= config["first_k_dense_replace"]):
            self.mlp = DeepseekMoE(dim, num_heads, mlp_ratio, dropout, layer_idx, history_steps, deepseek_residual, \
                router_init_weight, router_init_bias, router_original_init, shared_expert_weight, deepseek_n_shared_experts)
        else:
            self.mlp = DeepseekMLP()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
            
    def forward(
        self,
        tgt,
        memory, # encoder output
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
        x_scene_encoder = None,
        layer = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """

        bs, R, M, D = tgt.shape
        
        tgt_route = tgt
        tgt_route = self.router(tgt_route,memory,tgt_key_padding_mask,memory_key_padding_mask,m_pos,)
                
        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt2 = self.norm1(tgt)
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask.repeat(M, 1)
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)
        tgt_valid = tgt_tmp[tgt_valid_mask]
        tgt2_valid = self.norm2(tgt_valid)
        tgt2_valid, _ = self.m2m_attn(
            tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid
        )
        tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
        tgt = torch.zeros_like(tgt_tmp)
        tgt[tgt_valid_mask] = tgt_valid
        
        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        tgt2 = self.cross_attn(
            tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        dist_prediction = None
        
        tgt2 = self.norm4(tgt)
        if layer == 0:
            tgt2 = self.mlp(tgt2, tgt_route.view(bs, R * M, D))
            scores = None
        else:
            tgt2, scores = self.mlp(tgt2, tgt_route.view(bs, R * M, D))
            
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)
        
        if scores is not None:
            scores = scores.reshape(bs, R, M, -1)

        return tgt, dist_prediction, scores


class PlanningDecoder(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
        history_steps = 21,
        yaw_constraint=False,
        cat_x=False,
        deepseek_num_experts_per_tok = None,
        deepseek_n_routed_experts = None,
        deepseek_scoring_func = None,
        deepseek_aux_loss_alpha = None,
        deepseek_seq_aux = None,
        deepseek_norm_topk_prob = None,
        deepseek_hidden_size = None,
        deepseek_intermediate_size = None,
        deepseek_moe_intermediate_size = None,
        deepseek_n_shared_experts = None,
        deepseek_first_k_dense_replaces = None,
        deepseek_road_balance_loss = True,
        deepseek_residual = False,
        use_av_query_for_displacement = False,
        router_init_weight = "False",
        router_init_bias = "False",
        router_original_init = False,
        epoch_total_iter = 1428,
        deepseek_road_balance_loss_weight = 1,
        ifreturn_logit = False,
        shared_expert_weight = 1.0,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.yaw_constraint = yaw_constraint
        self.cat_x = cat_x
        self.router_init_weight = router_init_weight
        self.router_init_bias = router_init_bias
        self.cur_iter = 0
        self.epoch_total_iter = epoch_total_iter
        self.shared_expert_weight = shared_expert_weight
        
        config["num_experts_per_tok"] = deepseek_num_experts_per_tok
        config["n_routed_experts"] = deepseek_n_routed_experts
        config["scoring_func"] = deepseek_scoring_func
        config["aux_loss_alpha"] = deepseek_aux_loss_alpha
        config["seq_aux"] = deepseek_seq_aux
        config["norm_topk_prob"] = deepseek_norm_topk_prob
        config["hidden_size"] = deepseek_hidden_size
        config["intermediate_size"] = deepseek_intermediate_size
        config["moe_intermediate_size"] = deepseek_moe_intermediate_size
        config["n_shared_experts"] = deepseek_n_shared_experts
        config["first_k_dense_replace"] = deepseek_first_k_dense_replaces
        config["deepseek_road_balance_loss"] = deepseek_road_balance_loss
        config["deepseek_road_balance_loss_weight"] = deepseek_road_balance_loss_weight
        config['ifreturn_logit'] = ifreturn_logit
        
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout, index, history_steps = history_steps, future_steps = future_steps, \
                                deepseek_residual=deepseek_residual,\
                                    use_av_query_for_displacement=use_av_query_for_displacement, \
                                            router_init_weight = router_init_weight, 
                                            router_init_bias = router_init_bias,
                                            router_original_init = router_original_init,
                                            shared_expert_weight = shared_expert_weight,
                                            deepseek_n_shared_experts = deepseek_n_shared_experts,
                                            )
                for index in range(decoder_depth)
            ]
        )

        self.r_pos_emb = FourierEmbedding(3, dim, 64)
        self.r_encoder = PointsEncoder(6, dim)

        self.q_proj = nn.Linear(2 * dim, dim)

        self.m_emb = nn.Parameter(torch.Tensor(1, 1, num_mode, dim))
        self.m_pos = nn.Parameter(torch.Tensor(1, num_mode, dim))

        if self.cat_x:
            self.cat_x_proj = nn.Linear(2 * dim, dim)

        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)

        nn.init.normal_(self.m_emb, mean=0.0, std=0.01)
        nn.init.normal_(self.m_pos, mean=0.0, std=0.01)

    def forward(self, data, enc_data):
        self.cur_iter += 1
            
        enc_emb = enc_data["enc_emb"]
        enc_key_padding_mask = enc_data["enc_key_padding_mask"]

        x_scene_encoder = None

        r_position = data["reference_line"]["position"]
        r_vector = data["reference_line"]["vector"]
        r_orientation = data["reference_line"]["orientation"]
        r_valid_mask = data["reference_line"]["valid_mask"]
        r_key_padding_mask = ~r_valid_mask.any(-1)
            
        r_feature = torch.cat(
            [
                r_position - r_position[..., 0:1, :2],
                r_vector,
                torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),
            ],
            dim=-1,
        )

        bs, R, P, C = r_feature.shape
        r_valid_mask = r_valid_mask.view(bs * R, P)
        r_feature = r_feature.reshape(bs * R, P, C)
        r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)

        r_pos = torch.cat([r_position[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
        r_emb = r_emb + self.r_pos_emb(r_pos)

        r_emb = r_emb.unsqueeze(2).repeat(1, 1, self.num_mode, 1)
        m_emb = self.m_emb.repeat(bs, R, 1, 1)

        q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))
        
        dist_predictions = []
        scores_list = []
        for layer, blk in enumerate(self.decoder_blocks):
            q, dist_prediction, scores = blk(
                q,
                enc_emb,
                tgt_key_padding_mask=r_key_padding_mask,
                memory_key_padding_mask=enc_key_padding_mask,
                m_pos=self.m_pos,
                x_scene_encoder = x_scene_encoder,
                layer = layer,
            )
            assert torch.isfinite(q).all()
            dist_predictions.append(dist_prediction)
            if scores is not None:
                scores_list.append(scores)

        if self.cat_x:
            x = enc_emb[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, R, self.num_mode, 1)
            q = self.cat_x_proj(torch.cat([q, x], dim=-1))

        loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc, yaw, vel], dim=-1)

        return traj, pi, None, None, None, None, None, None, dist_predictions, scores_list