import torch
import torch.nn as nn

from ..layers.common_layers import build_mlp
from ..layers.embedding import NATSequenceEncoder

import math


class AgentEncoder(nn.Module):
    def __init__(
        self,
        state_channel=6,
        history_channel=9,
        dim=128,
        hist_steps=21,
        use_ego_history=False,
        drop_path=0.2,
        state_attn_encoder=True,
        state_dropout=0.75,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.state_channel = state_channel
        self.use_ego_history = use_ego_history
        self.hist_steps = hist_steps
        self.state_attn_encoder = state_attn_encoder

        self.history_encoder = NATSequenceEncoder(
            in_chans=history_channel, embed_dim=dim // 4, drop_path_rate=drop_path
        )

        if not use_ego_history:
            if not self.state_attn_encoder:
                self.ego_state_emb = build_mlp(state_channel, [dim] * 2, norm="bn")
            else:
                self.ego_state_emb = StateAttentionEncoder(
                    state_channel, dim, state_dropout
                )

        self.type_emb = nn.Embedding(4, dim)

    @staticmethod
    def to_vector(feat, valid_mask):
        vec_mask = valid_mask[..., :-1] & valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)

        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, data, cur_temp_cons = 0, AV_pos=None, AV_heading=None):
        if cur_temp_cons != 0:
            Ts = cur_temp_cons
            Te = cur_temp_cons + self.hist_steps
            position = data["agent"]["position"][:, :, Ts:Te].clone() #torch.Size([B, N+1, History_step, 2])
            heading = data["agent"]["heading"][:, :, Ts:Te].clone()
            velocity = data["agent"]["velocity"][:, :, Ts:Te].clone()
            shape = data["agent"]["shape"][:, :, Ts:Te].clone()
            category = data["agent"]["category"].long().clone()
            valid_mask = data["agent"]["valid_mask"][:, :, Ts:Te].clone()
            
            cos_h = torch.cos(AV_heading)[:, None, None]
            sin_h = torch.sin(AV_heading)[:, None, None]
            rotate_mat = torch.cat([
                torch.cat([cos_h, -sin_h], dim=-1),  
                torch.cat([sin_h, cos_h], dim=-1)  
            ], dim=-2)
                
            position = torch.matmul(position - AV_pos[:, None, None], rotate_mat[:, None])
            velocity = torch.matmul(velocity, rotate_mat[:, None])
            heading -= AV_heading[:, None, None]
        else:
            T = self.hist_steps
            position = data["agent"]["position"][:, :, :T] #torch.Size([B, N+1, History_step, 2])
            heading = data["agent"]["heading"][:, :, :T]
            velocity = data["agent"]["velocity"][:, :, :T]
            shape = data["agent"]["shape"][:, :, :T]
            category = data["agent"]["category"].long()
            valid_mask = data["agent"]["valid_mask"][:, :, :T]
                
        heading_vec = self.to_vector(heading, valid_mask) #torch.Size([B, N+1, History_step -1])
        valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1] #torch.Size([B, N+1, History_step -1])
        agent_feature = torch.cat( #torch.Size([B, N+1, History_step -1, 9])
            [
                self.to_vector(position, valid_mask),
                self.to_vector(velocity, valid_mask),
                torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
                shape[:, :, 1:],
                valid_mask_vec.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        
        bs, A, T, _ = agent_feature.shape #2, 49, 20
        agent_feature = agent_feature.view(bs * A, T, -1)

        valid_agent_mask = valid_mask.any(-1).flatten()

        x_agent_tmp = self.history_encoder( #torch.Size([B*Valid Agent, 128])
            agent_feature[valid_agent_mask].permute(0, 2, 1).contiguous() #torch.Size([Valid, state, History_step -1])
        )
        x_agent = torch.zeros(bs * A, self.dim, device=position.device)
        x_agent[valid_agent_mask] = x_agent_tmp
        x_agent = x_agent.view(bs, A, self.dim) #torch.Size([B, N+1, 128])

        if not self.use_ego_history:
            if "current_state_timestep" in data:
                ego_feature = data["current_state_timestep"][cur_temp_cons][:, : self.state_channel] #x, y, heading, vel, acc, steering_angle, yaw_rate
                # ego_feature[:, :3] = 0 #전처리에서 하도록 수정함
            else:
                ego_feature = data["current_state"][:, : self.state_channel] #x, y, heading, vel, acc, steering_angle, yaw_rate
            x_ego = self.ego_state_emb(ego_feature)
            x_agent[:, 0] = x_ego

        x_type = self.type_emb(category)

        return x_agent + x_type


class StateAttentionEncoder(nn.Module):
    def __init__(self, state_channel, dim, state_dropout=0.5) -> None:
        super().__init__()

        self.state_channel = state_channel
        self.state_dropout = state_dropout
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(state_channel)])
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_channel, dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, dim))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x):
        x_embed = []
        for i, linear in enumerate(self.linears):
            x_embed.append(linear(x[:, i, None]))
        x_embed = torch.stack(x_embed, dim=1)
        pos_embed = self.pos_embed.repeat(x_embed.shape[0], 1, 1)
        x_embed = x_embed + pos_embed

        if self.training and self.state_dropout > 0:
            visible_tokens = torch.zeros(
                (x_embed.shape[0], 3), device=x.device, dtype=torch.bool
            )
            dropout_tokens = (
                torch.rand((x_embed.shape[0], self.state_channel - 3), device=x.device)
                < self.state_dropout
            )
            key_padding_mask = torch.concat([visible_tokens, dropout_tokens], dim=1)
        else:
            key_padding_mask = None

        query = self.query.repeat(x_embed.shape[0], 1, 1)

        x_state = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )[0]

        return x_state[:, 0]



class AgentEncoder_TemporalConsis_Batch(nn.Module):
    def __init__(
        self,
        state_channel=6,
        history_channel=9,
        dim=128,
        hist_steps=21,
        use_ego_history=False,
        drop_path=0.2,
        state_attn_encoder=True,
        state_dropout=0.75,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.state_channel = state_channel
        self.use_ego_history = use_ego_history
        self.hist_steps = hist_steps
        self.state_attn_encoder = state_attn_encoder

        self.history_encoder = NATSequenceEncoder(
            in_chans=history_channel, embed_dim=dim // 4, drop_path_rate=drop_path
        )

        if not use_ego_history:
            if not self.state_attn_encoder:
                self.ego_state_emb = build_mlp(state_channel, [dim] * 2, norm="bn")
            else:
                self.ego_state_emb = StateAttentionEncoder(
                    state_channel, dim, state_dropout
                )

        self.type_emb = nn.Embedding(4, dim)

    @staticmethod
    def to_vector(feat, valid_mask):
        vec_mask = valid_mask[..., :-1] & valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)

        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, data, AV_pos=None, TC_idcs=None):
        if AV_pos is not None :
            range_indices = torch.arange(21).unsqueeze(0).to(TC_idcs) + TC_idcs.unsqueeze(1)  # shape: [len(Ts), 21]
            position = data["agent"]["position"][:, :, range_indices] #torch.Size([B, N+1, History_step, 2])
            heading = data["agent"]["heading"][:, :, range_indices]
            velocity = data["agent"]["velocity"][:, :, range_indices]
            shape = data["agent"]["shape"][:, :, range_indices]
            category = data["agent"]["category"].long()[:,:,None].repeat(1,1,len(TC_idcs))
            valid_mask = data["agent"]["valid_mask"][:, :, range_indices]
            
            position = position - AV_pos[:,None,:,None]
            
            position = position.transpose(2,1).flatten(0,1)
            heading = heading.transpose(2,1).flatten(0,1)
            velocity = velocity.transpose(2,1).flatten(0,1)
            shape = shape.transpose(2,1).flatten(0,1)
            category = category.transpose(2,1).flatten(0,1)
            valid_mask = valid_mask.transpose(2,1).flatten(0,1)
            
        else:
            T = self.hist_steps
            position = data["agent"]["position"][:, :, :T] #torch.Size([B, N+1, History_step, 2])
            heading = data["agent"]["heading"][:, :, :T]
            velocity = data["agent"]["velocity"][:, :, :T]
            shape = data["agent"]["shape"][:, :, :T]
            category = data["agent"]["category"].long()
            valid_mask = data["agent"]["valid_mask"][:, :, :T]

        heading_vec = self.to_vector(heading, valid_mask) #torch.Size([B, N+1, History_step -1])
        valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1] #torch.Size([B, N+1, History_step -1])
        agent_feature = torch.cat( #torch.Size([B, N+1, History_step -1, 9])
            [
                self.to_vector(position, valid_mask),
                self.to_vector(velocity, valid_mask),
                torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
                shape[:, :, 1:],
                valid_mask_vec.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        bs, A, T, _ = agent_feature.shape #2, 49, 20
        agent_feature = agent_feature.view(bs * A, T, -1)
        valid_agent_mask = valid_mask.any(-1).flatten()

        x_agent_tmp = self.history_encoder( #torch.Size([B*Valid Agent, 128])
            agent_feature[valid_agent_mask].permute(0, 2, 1).contiguous() #torch.Size([Valid, state, History_step -1])
        )
        x_agent = torch.zeros(bs * A, self.dim, device=position.device)
        x_agent[valid_agent_mask] = x_agent_tmp
        x_agent = x_agent.view(bs, A, self.dim) #torch.Size([B, N+1, 128])

        if not self.use_ego_history:
            if AV_pos is not None :
                ego_feature = data["current_state"][:, : self.state_channel][:,None].repeat(1,len(TC_idcs),1).flatten(0,1) #x, y, heading, vel, acc, steering_angle, yaw_rate
                x_ego = self.ego_state_emb(ego_feature)
                x_agent[:, 0] = x_ego
            else:
                ego_feature = data["current_state"][:, : self.state_channel] #x, y, heading, vel, acc, steering_angle, yaw_rate
                x_ego = self.ego_state_emb(ego_feature)
                x_agent[:, 0] = x_ego

        x_type = self.type_emb(category)

        return x_agent + x_type

