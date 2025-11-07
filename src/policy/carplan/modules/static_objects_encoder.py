import math
import torch
import torch.nn as nn

from ..layers.fourier_embedding import FourierEmbedding
from ..layers.embedding import NATSequenceEncoder

class StaticObjectsDispEncoder(nn.Module):
    def __init__(self, dim, hist_steps=False,) -> None:
        super().__init__()

        self.hist_steps = hist_steps
        
        self.obj_encoder = FourierEmbedding(2, dim, 64)
        self.type_emb = nn.Embedding(4, dim)
        
        self.history_encoder = NATSequenceEncoder(
            in_chans=4, embed_dim=dim // 4, drop_path_rate=0.2
        )

        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.01)

    def forward(self, data, cur_temp_cons = 0, AV_pos=None, AV_cur_heading=None, AV_cur_vel=None) -> torch.Tensor:
        pos = data["static_objects"]["position"].clone()
        heading = data["static_objects"]["heading"].clone()
        shape = data["static_objects"]["shape"]
        category = data["static_objects"]["category"].long()
        valid_mask = data["static_objects"]["valid_mask"]  # [bs, N]
        
        AV_position = data["agent"]["position"][:, :1, :self.hist_steps]
        AV_heading = data["agent"]["heading"][:, :1, :self.hist_steps]
        
        # Time Step별
        point_position = pos.unsqueeze(-2).repeat(1, 1, self.hist_steps, 1) - AV_position # 아 ABS 씌워야 하나.. 모르겠네
        point_orientation = heading.unsqueeze(-1).repeat(1, 1, self.hist_steps) - AV_heading
        
        static_feature = torch.cat(
            [
                point_position,
                torch.stack(
                    [
                        point_orientation.cos(),
                        point_orientation.sin(),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        
        bs, S, T, C = static_feature.shape
        
        if S == 0:
            pass
        else:
            valid_mask = valid_mask
            polygon_feature_tmp = self.history_encoder( #torch.Size([B*Valid Agent, 128])
                static_feature[valid_mask].permute(0, 2, 1).contiguous() #torch.Size([Valid, state, History_step -1])
            )
            x_static_ = torch.zeros(bs, S, polygon_feature_tmp.shape[-1], device=point_position.device)
            x_static_[valid_mask] = polygon_feature_tmp
            x_static = x_static_ #.view(bs, M, P, self.dim) #torch.Size([B, N+1, 128])   
        
            
        obj_emb_tmp = self.obj_encoder(shape) + self.type_emb(category.long())
        obj_emb = torch.zeros_like(obj_emb_tmp)
        obj_emb[valid_mask] = obj_emb_tmp[valid_mask]

        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        obj_pos = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

        if S == 0:
            return obj_emb
        else:
            return x_static + obj_emb

class StaticObjectsEncoder(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.obj_encoder = FourierEmbedding(2, dim, 64)
        self.type_emb = nn.Embedding(4, dim)

        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.01)

    def forward(self, data, cur_temp_cons = 0, AV_pos=None, AV_cur_heading=None, AV_cur_vel=None) -> torch.Tensor:
        pos = data["static_objects"]["position"].clone()
        heading = data["static_objects"]["heading"].clone()
        shape = data["static_objects"]["shape"]
        category = data["static_objects"]["category"].long()
        valid_mask = data["static_objects"]["valid_mask"]  # [bs, N]
        
        if cur_temp_cons != 0 :
            cos_h = torch.cos(AV_cur_heading)[:, None, None]
            sin_h = torch.sin(AV_cur_heading)[:, None, None]
            rotate_mat = torch.cat([
                torch.cat([cos_h, -sin_h], dim=-1),  
                torch.cat([sin_h, cos_h], dim=-1)  
            ], dim=-2)
            
            pos = torch.matmul(pos - AV_pos[:, None], rotate_mat)
            heading -= AV_cur_heading[:, None]
            # heading = (heading + math.pi) % (2 * math.pi) - math.pi
            
        obj_emb_tmp = self.obj_encoder(shape) + self.type_emb(category.long())
        obj_emb = torch.zeros_like(obj_emb_tmp)
        obj_emb[valid_mask] = obj_emb_tmp[valid_mask]

        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        obj_pos = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

        return obj_emb, obj_pos, ~valid_mask


class StaticObjectsEncoder_TemporalConsis_Batch(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.obj_encoder = FourierEmbedding(2, dim, 64)
        self.type_emb = nn.Embedding(4, dim)

        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.01)

    def forward(self, data, AV_pos = None):
        if AV_pos is not None:
            pos = data["static_objects"]["position"][:,None].repeat(1,AV_pos.shape[1],1,1)
            heading = data["static_objects"]["heading"][:,None].repeat(1,AV_pos.shape[1],1)
            shape = data["static_objects"]["shape"][:,None].repeat(1,AV_pos.shape[1],1,1)
            category = data["static_objects"]["category"].long()[:,None].repeat(1,AV_pos.shape[1],1)
            valid_mask = data["static_objects"]["valid_mask"][:,None].repeat(1,AV_pos.shape[1],1)  # [bs, N]
            
            pos = pos - AV_pos[:,:,None]
            
            pos = pos.flatten(0,1)
            heading = heading.flatten(0,1)
            shape = shape.flatten(0,1)
            category = category.flatten(0,1)
            valid_mask = valid_mask.flatten(0,1)
        else:
            pos = data["static_objects"]["position"]
            heading = data["static_objects"]["heading"]
            shape = data["static_objects"]["shape"]
            category = data["static_objects"]["category"].long()
            valid_mask = data["static_objects"]["valid_mask"]  # [bs, N]

        obj_emb_tmp = self.obj_encoder(shape) + self.type_emb(category.long())
        obj_emb = torch.zeros_like(obj_emb_tmp)
        obj_emb[valid_mask] = obj_emb_tmp[valid_mask]

        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        obj_pos = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

        return obj_emb, obj_pos, ~valid_mask
