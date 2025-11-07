import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding
from ..layers.embedding import NATSequenceEncoder


import math                

class MapDispEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
        hist_steps=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        self.hist_steps = hist_steps
        
        self.polygon_channel = (
            polygon_channel + 4 if use_lane_boundary else polygon_channel
        )

        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)
        
        self.history_encoder = NATSequenceEncoder(
            in_chans=4, embed_dim=dim // 4, drop_path_rate=0.2
        )

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data, cur_temp_cons = 0, AV_pos=None, AV_cur_heading=None, AV_cur_vel=None) -> torch.Tensor:
        
        polygon_center = data["map"]["polygon_center"].clone()
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"].clone()
        point_vector = data["map"]["point_vector"].clone()
        point_orientation = data["map"]["point_orientation"].clone()
        valid_mask = data["map"]["valid_mask"]
        
        AV_position = data["agent"]["position"][:, :1, :self.hist_steps]
        AV_heading = data["agent"]["heading"][:, :1, :self.hist_steps]
        
        # Time Step별
        point_position = polygon_center[..., :2].unsqueeze(-2).repeat(1, 1, self.hist_steps, 1) - AV_position # 아 ABS 씌워야 하나.. 모르겠네
        point_orientation = polygon_center[..., 2].unsqueeze(-1).repeat(1, 1, self.hist_steps) - AV_heading

        polygon_feature = torch.cat(
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

        bs, M, T, C = polygon_feature.shape
        
        valid_mask = valid_mask.any(-1)
        polygon_feature_tmp = self.history_encoder( #torch.Size([B*Valid Agent, 128])
            polygon_feature[valid_mask].permute(0, 2, 1).contiguous() #torch.Size([Valid, state, History_step -1])
        )
        x_polygon_ = torch.zeros(bs, M, self.dim, device=point_position.device)
        x_polygon_[valid_mask] = polygon_feature_tmp
        x_polygon = x_polygon_ #.view(bs, M, P, self.dim) #torch.Size([B, N+1, 128])        

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon = x_polygon + x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
    
class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        self.polygon_channel = (
            polygon_channel + 4 if use_lane_boundary else polygon_channel
        )

        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data, cur_temp_cons = 0, AV_pos=None, AV_cur_heading=None, AV_cur_vel=None) -> torch.Tensor:
        
        polygon_center = data["map"]["polygon_center"].clone()
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"].clone()
        point_vector = data["map"]["point_vector"].clone()
        point_orientation = data["map"]["point_orientation"].clone()
        valid_mask = data["map"]["valid_mask"]

        if cur_temp_cons != 0 :
            cos_h = torch.cos(AV_cur_heading)[:, None, None]
            sin_h = torch.sin(AV_cur_heading)[:, None, None]
            rotate_mat = torch.cat([
                torch.cat([cos_h, -sin_h], dim=-1),  
                torch.cat([sin_h, cos_h], dim=-1)  
            ], dim=-2)
            
            point_position = torch.matmul(point_position - AV_pos[:, None, None, None], rotate_mat[:, None, None])
            point_vector = torch.matmul(point_vector, rotate_mat[:, None, None])
            point_orientation -= AV_cur_heading[:, None, None, None]
            polygon_center[...,:2] = torch.matmul(polygon_center[...,:2] - AV_pos[:, None], rotate_mat)
            polygon_center[..., 2] -= AV_cur_heading[:, None]
            # point_orientation = (point_orientation + math.pi) % (2 * math.pi) - math.pi
            # polygon_center[..., 2] = (polygon_center[..., 2] + math.pi) % (2 * math.pi) - math.pi

        if self.use_lane_boundary:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                    point_position[:, :, 1] - point_position[:, :, 0],
                    point_position[:, :, 2] - point_position[:, :, 0],
                ],
                dim=-1,
            )
        else:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1) #torch.Size([B, M, 128])

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon = x_polygon + x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon



class MapEncoder_TemporalConsis_Batch(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        self.polygon_channel = (
            polygon_channel + 4 if use_lane_boundary else polygon_channel
        )

        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data, AV_pos=None) -> torch.Tensor:
        
        polygon_center = data["map"]["polygon_center"]
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"]
        point_vector = data["map"]["point_vector"]
        point_orientation = data["map"]["point_orientation"]
        valid_mask = data["map"]["valid_mask"]

        if AV_pos is not None:
            polygon_center = data["map"]["polygon_center"][:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            polygon_type = data["map"]["polygon_type"].long()[:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            polygon_on_route = data["map"]["polygon_on_route"].long()[:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            polygon_tl_status = data["map"]["polygon_tl_status"].long()[:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"][:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            polygon_speed_limit = data["map"]["polygon_speed_limit"][:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            point_position = data["map"]["point_position"][:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            point_vector = data["map"]["point_vector"][:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            point_orientation = data["map"]["point_orientation"][:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            valid_mask = data["map"]["valid_mask"][:,None].repeat_interleave(AV_pos.shape[1], dim=1) 
            
            polygon_center[...,:2] = polygon_center[...,:2] - AV_pos[:,:,None]
            point_position = point_position - AV_pos[:,:,None,None,None]
            
            polygon_center = polygon_center.flatten(0,1)
            polygon_type = polygon_type.flatten(0,1)
            polygon_on_route = polygon_on_route.flatten(0,1)
            polygon_tl_status = polygon_tl_status.flatten(0,1)
            polygon_has_speed_limit = polygon_has_speed_limit.flatten(0,1)
            polygon_speed_limit = polygon_speed_limit.flatten(0,1)
            point_position = point_position.flatten(0,1)
            point_vector = point_vector.flatten(0,1)
            point_orientation = point_orientation.flatten(0,1)
            valid_mask = valid_mask.flatten(0,1)
            
        else:
            polygon_center = data["map"]["polygon_center"]
            polygon_type = data["map"]["polygon_type"].long()
            polygon_on_route = data["map"]["polygon_on_route"].long()
            polygon_tl_status = data["map"]["polygon_tl_status"].long()
            polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
            polygon_speed_limit = data["map"]["polygon_speed_limit"]
            point_position = data["map"]["point_position"]
            point_vector = data["map"]["point_vector"]
            point_orientation = data["map"]["point_orientation"]
            valid_mask = data["map"]["valid_mask"]

        if self.use_lane_boundary:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                    point_position[:, :, 1] - point_position[:, :, 0],
                    point_position[:, :, 2] - point_position[:, :, 0],
                ],
                dim=-1,
            )
        else:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1) #torch.Size([B, M, 128])

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon = x_polygon + x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon



class MapEncoder_w_inverse_traffic(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        self.polygon_channel = (
            polygon_channel + 4 if use_lane_boundary else polygon_channel
        )

        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data) -> torch.Tensor:
        polygon_center = data["map"]["polygon_center"]
        polygon_type = data["map"]["polygon_type"].long()# lane->0, lane_connector->1, Crosswalk->2
        polygon_on_route = data["map"]["polygon_on_route"].long() 
        polygon_tl_status = data["map"]["polygon_tl_status"].long() # Green -> 0, Red -> 2,  Unknown=3, Yellow=1
        polygon_tl_status_inverse = polygon_tl_status.clone()
        polygon_tl_status_inverse[torch.logical_and(polygon_tl_status == 2, polygon_type == 1)] = 0 # Red -> Green
        polygon_tl_status_inverse[torch.logical_and(polygon_tl_status == 1, polygon_type == 1)] = 0 # Yellow -> Green
        polygon_tl_status_inverse[torch.logical_and(polygon_tl_status == 0, polygon_type == 1)] = 2 # Green -> Red
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"]
        point_vector = data["map"]["point_vector"]
        point_orientation = data["map"]["point_orientation"]
        valid_mask = data["map"]["valid_mask"]

        if self.use_lane_boundary:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                    point_position[:, :, 1] - point_position[:, :, 0],
                    point_position[:, :, 2] - point_position[:, :, 0],
                ],
                dim=-1,
            )
        else:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        with torch.no_grad():
            x_inverse_tl_status = self.traffic_light_emb(polygon_tl_status_inverse)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        
        # # 대체 방법: 텐서 연산으로 결과 생성
        # updated_x_speed_limit = torch.where(
        #     ~polygon_has_speed_limit.unsqueeze(-1),  # 마스크 확장
        #     self.unknown_speed_emb.weight.expand_as(x_speed_limit),  # 대체 값
        #     x_speed_limit  # 기존 값
        # )
        # x_speed_limit = updated_x_speed_limit

        
        # x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight
        x_polygon_inverse_traffic = x_polygon.clone()
        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit
        with torch.no_grad():
            x_polygon_inverse_traffic += x_type + x_on_route + x_inverse_tl_status + x_speed_limit
        


        return x_polygon, x_polygon_inverse_traffic