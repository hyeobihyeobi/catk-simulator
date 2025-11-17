from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding
from ..layers.mlp_layer import MLPLayer


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

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
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape

        unvalid_batch_mask = tgt_key_padding_mask.all(-1)
        tgt_tmp = tgt[~unvalid_batch_mask]
        tgt_tmp = tgt_tmp.transpose(1, 2).reshape((bs-unvalid_batch_mask.sum()) * M, R, D)
        tgt2 = self.norm1(tgt_tmp)
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask[~unvalid_batch_mask].repeat(M, 1)
        )[0]
        # if torch.isnan(tgt2).any():
        #     print('eher')
        # tgt2 = torch.nan_to_num(tgt2, nan=0.0)
        tgt = torch.zeros_like(tgt)
        tgt_tmp = tgt_tmp + self.dropout1(tgt2)
        tgt[~unvalid_batch_mask] = tgt_tmp.reshape((bs-unvalid_batch_mask.sum()), M, R, D).transpose(1, 2)

        tgt_tmp = tgt.reshape(bs * R, M, D)
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
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)

        return tgt


class PlanningDecoder_origin(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
        yaw_constraint=False,
        cat_x=False,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.yaw_constraint = yaw_constraint
        self.cat_x = cat_x

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
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
        enc_emb = enc_data["enc_emb"]
        enc_key_padding_mask = enc_data["enc_key_padding_mask"]
        
        try:
            cur_idx = enc_data["cur_idx"]
            AV_cur_xy = enc_data["AV_cur_xy"]
            AV_cur_heading = enc_data["AV_cur_heading"]
        except:
            cur_idx, AV_cur_xy, AV_cur_heading = 0, None, None

        r_position = data["position"]
        r_vector = data["vector"]
        r_orientation = data["orientation"]
        r_valid_mask = data["valid_mask"]
        r_key_padding_mask = ~r_valid_mask.any(-1)
        
        if cur_idx != 0 :
            cos_h = torch.cos(AV_cur_heading)[:, None, None]
            sin_h = torch.sin(AV_cur_heading)[:, None, None]
            rotate_mat = torch.cat([
                torch.cat([cos_h, -sin_h], dim=-1),  
                torch.cat([sin_h, cos_h], dim=-1)  
            ], dim=-2)
            
            r_position = torch.matmul(r_position - AV_cur_xy[:, None, None], rotate_mat[:, None])
            r_vector = torch.matmul(r_vector, rotate_mat[:, None])
            r_orientation -= AV_cur_heading[:, None, None]

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

        for blk in self.decoder_blocks:
            q = blk(
                q,
                enc_emb,
                tgt_key_padding_mask=r_key_padding_mask,
                memory_key_padding_mask=enc_key_padding_mask,
                m_pos=self.m_pos,
            )
            assert torch.isfinite(q).all()

        if self.cat_x:
            x = enc_emb[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, R, self.num_mode, 1)
            q = self.cat_x_proj(torch.cat([q, x], dim=-1))

        loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc, yaw, vel], dim=-1)

        return traj, pi, None, None, None, None, None, None, None, None