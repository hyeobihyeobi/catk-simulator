from copy import deepcopy
import math
import numpy as np
import numpy.typing as npt
from scipy.special import softmax

import torch
import torch.nn as nn

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor, DistPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder_deepseek_v2 import PlanningDecoder as deepseek_decoder
from .layers.mlp_layer import MLPLayer


class PlanningModel_MoE(nn.Module):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        radius: float = 80.0,
        use_collision_loss = True,
        use_ref_line = True,
        is_simulation=False,
        important_loss_weight = 0.0,
        load_loss_weight = 0.0,
        moe_CIL = False,
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
        deepseek_road_balance_loss=True,
        deepseek_residual=False,
        distance_loss_weight=False,
        use_av_query_for_displacement = False,
        av_cat = False,
        collision_loss_weight = 1.0,
        cil_loss_weight = 1.0,
        planning_loss_weight = 1.0,
        router_init_weight = "False",
        router_init_bias = "False",
        router_original_init = False,
        epoch_total_iter = 1428,
        prediction_loss_weight = 1.0,
        deepseek_road_balance_loss_weight = 1,
        displcement_loss_weight = 1.0,
        ifreturn_logit = False,
        no_displacement_for_CLSR = False,
        no_prediction_for_CLSR = False,
        shared_expert_weight = 1.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = radius
        self.ref_free_traj = ref_free_traj
        self.use_collision_loss = use_collision_loss
        self.use_ref_line = use_ref_line
        self.is_simulation = is_simulation
        self.important_loss_weight = important_loss_weight
        self.load_loss_weight = load_loss_weight
        self.moe_CIL = moe_CIL
        self.distance_loss_weight = distance_loss_weight
        self.av_cat = av_cat
        self.collision_loss_weight = collision_loss_weight
        self.cil_loss_weight = cil_loss_weight
        self.planning_loss_weight = planning_loss_weight
        self.prediction_loss_weight = prediction_loss_weight
        self.displcement_loss_weight = displcement_loss_weight
        
        self.no_displacement_for_CLSR = no_displacement_for_CLSR
        self.no_prediction_for_CLSR = no_prediction_for_CLSR
        
        if self.moe_CIL or self.use_hidden_proj:
            self.use_contrast_loss = True
        else:
            self.use_contrast_loss = False
        
        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )
        
        self.distance_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )
            
        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.displacement_encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.displacement_norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.dist_predictor = DistPredictor(dim=dim, future_steps=future_steps)
        
        if self.av_cat:
            self.av_cat_x_proj = nn.Linear(2 * dim, dim)
        
        self.planning_decoder = deepseek_decoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
            deepseek_num_experts_per_tok = deepseek_num_experts_per_tok,
            deepseek_n_routed_experts = deepseek_n_routed_experts,
            deepseek_scoring_func = deepseek_scoring_func,
            deepseek_aux_loss_alpha = deepseek_aux_loss_alpha,
            deepseek_seq_aux = deepseek_seq_aux,
            deepseek_norm_topk_prob = deepseek_norm_topk_prob,
            deepseek_hidden_size = deepseek_hidden_size,
            deepseek_intermediate_size = deepseek_intermediate_size,
            deepseek_moe_intermediate_size = deepseek_moe_intermediate_size,
            deepseek_n_shared_experts = deepseek_n_shared_experts,
            deepseek_first_k_dense_replaces = deepseek_first_k_dense_replaces,
            deepseek_road_balance_loss = deepseek_road_balance_loss,
            deepseek_residual = deepseek_residual,
            use_av_query_for_displacement = use_av_query_for_displacement,
            router_init_weight = router_init_weight,
            router_init_bias = router_init_bias,
            router_original_init = router_original_init,
            deepseek_road_balance_loss_weight = deepseek_road_balance_loss_weight,
            epoch_total_iter = epoch_total_iter,
            ifreturn_logit = ifreturn_logit,
            shared_expert_weight = shared_expert_weight,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )
            if moe_CIL:
                self.hidden_proj_q = nn.Sequential(
                    nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
                )
                self.hidden_proj_tgt_route = nn.Sequential(
                    nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
                )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _trim_candidates(
        self,
        candidate_trajectories: np.ndarray,
        probability: np.ndarray,
        ego_pos: torch.Tensor,
        ego_heading: torch.Tensor,
        ref_free_trajectory: np.ndarray = None,
    ) -> npt.NDArray[np.float32]:
        """
        candidate_trajectories: (n_ref, n_mode, 80, 3)
        probability: (n_ref, n_mode)
        """
        if len(candidate_trajectories.shape) == 4:
            n_ref, n_mode, T, C = candidate_trajectories.shape
            candidate_trajectories = candidate_trajectories.reshape(-1, T, C)
            probability = probability.reshape(-1)

        sorted_idx = np.argsort(-probability)
        sorted_candidate_trajectories = candidate_trajectories[sorted_idx][: self._topk]
        sorted_probability = probability[sorted_idx][: self._topk]
        sorted_probability = softmax(sorted_probability)

        if ref_free_trajectory is not None:
            sorted_candidate_trajectories = np.concatenate(
                [sorted_candidate_trajectories, ref_free_trajectory[None, ...]],
                axis=0,
            )
            sorted_probability = np.concatenate([sorted_probability, [0.25]], axis=0)

        # to global
        origin = ego_pos
        angle = ego_heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        sorted_candidate_trajectories[..., :2] = (
            np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        sorted_candidate_trajectories = np.concatenate(
            [sorted_candidate_trajectories[..., 0:1, :], sorted_candidate_trajectories],
            axis=-2,
        )

        return sorted_candidate_trajectories, sorted_probability

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1] #torch.Size([B, N+1, 2]) #AV 포함 #AV-centric #현재타임스텝 위치
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1] #torch.Size([B, N+1])
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps] #torch.Size([B, N+1, History_step])
        polygon_center = data["map"]["polygon_center"] #torch.Size([B, M, 3])
        polygon_mask = data["map"]["valid_mask"] #torch.Size([B, M, 20])

        bs, A = agent_pos.shape[0:2] #2, 49

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1) #torch.Size([B, N+1+M, 2])
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1) #torch.Size([B, N+1+M])
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1) #torch.Size([B, N+1+M, 3])

        agent_key_padding = ~(agent_mask.any(-1)) # masking for transformer attention(not to see) #torch.Size([B, N+1])
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1) #torch.Size([B, N+1+M])

        x_agent = self.agent_encoder(data) #torch.Size([B, N+1, 128])
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        M_shape, S_shape = x_polygon.shape[1], x_static.shape[1]

        x = torch.cat([x_agent, x_polygon, x_static], dim=1) #Agent-centric feature

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos) 

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed #Agent-centric feature에 position 정보 추가

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x) #torch.Size([B, N+1+M+S, 128])

        if self.no_prediction_for_CLSR:
            prediction = torch.zeros(bs, A-1, self.future_steps, 6).to(x)
        else:
            prediction = self.agent_predictor(x[:, 1:A]) #torch.Size([B, N, Future_step, 6])

        x_scene_encoder = None
            
        for dl in self.displacement_encoder_blocks:
            x = dl(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.displacement_norm(x)

        if self.is_simulation:
            pass
        else:
            if self.av_cat:
                cat_av_dist = torch.cat([x[:, 1:], x[:, 0:1].repeat(1, A+M_shape+S_shape-1, 1)], dim=-1)
                cat_av_dist = self.av_cat_x_proj(cat_av_dist)
                dist_prediction = self.dist_predictor(cat_av_dist)
            else:
                dist_prediction = self.dist_predictor(x[:, 1:])

        ref_line_available = data["reference_line"]["position"].shape[1] > 0
        R, M = data["reference_line"]["position"].shape[1], 12
        
        if ref_line_available:
            trajectory, probability, pred_scenario_type, q, tgt_route, gates, load, gates_dict, dist_predictions, scores_list = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask, "enc_pos": pos, "x_scene_encoder": x_scene_encoder}
            )
        else:
            trajectory, probability, pred_scenario_type, q, tgt_route, gates, load, gates_dict,scores_list = None, None, None, None, None, None, None, None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "pred_scenario_type": pred_scenario_type,
            "prediction": prediction,  # (bs, A-1, T, 2)
            "load": load,
            "gates": gates,
            "gates_dict": gates_dict,
            "agent_pos": agent_pos,
            "scores_list": scores_list,
        }
        
        if self.is_simulation:
            pass
        else:
            out["dist_prediction"] = dist_prediction
                
        if not self.is_simulation:
            if self.no_prediction_for_CLSR:
                pass
            elif self.no_displacement_for_CLSR == "all":
                pass
            else:
                AV_future = data["agent"]["position"][:, :1, self.history_steps:self.history_steps+self.future_steps]
                Agent_future = data["agent"]["position"][:, 1:, self.history_steps:self.history_steps+self.future_steps] # AV Centric future
                Map_future = data["map"]["polygon_center"][:, :, :2].unsqueeze(-2).repeat(1, 1, self.future_steps, 1)
                Static_future = data["static_objects"]["position"].unsqueeze(-2).repeat(1, 1, self.future_steps, 1)
                
                if self.no_displacement_for_CLSR == "agent":
                    x_future = torch.cat([Map_future, Static_future], dim=1)
                else:
                    x_future = torch.cat([Agent_future, Map_future, Static_future], dim=1)
                
                dist_prediction_GT = x_future-AV_future
                dist_prediction_GT_mask = data["agent"]["valid_mask"][:, :1, self.history_steps:self.history_steps+self.future_steps] * data["agent"]["valid_mask"][:, 1:, self.history_steps:self.history_steps+self.future_steps]
                
                dist_prediction_GT_Dist = torch.norm(dist_prediction_GT, dim=-1)
                x_diff = (dist_prediction_GT[..., 0:1]) #.unsqueeze(-1).type(torch.float32)  # [1296, 48, 1]
                y_diff = (dist_prediction_GT[..., 1:2]) #.unsqueeze(-1).type(torch.float32)  # [1296, 48, 1]
                rot_prediction_GT = torch.atan2(y_diff.squeeze(-1), x_diff.squeeze(-1)) #.unsqueeze(-1).type(torch.float32)  # [1296, 48, 1]

                out["dist_prediction_GT"] = torch.stack((dist_prediction_GT_Dist, rot_prediction_GT), dim=-1)
                if self.no_displacement_for_CLSR == "agent":
                    out["dist_prediction_mask"] = torch.cat([polygon_key_padding.unsqueeze(-1).repeat(1, 1, self.future_steps), \
                        static_key_padding.unsqueeze(-1).repeat(1, 1, self.future_steps)], dim=1)
                else:
                    out["dist_prediction_mask"] = torch.cat([dist_prediction_GT_mask, \
                        (~polygon_key_padding).unsqueeze(-1).repeat(1, 1, self.future_steps), (~static_key_padding).unsqueeze(-1).repeat(1, 1, self.future_steps)], dim=1)

                pred_AV_centric = prediction[...,:2] + data["agent"]["position"][:, 1:, self.history_steps - 1][:,:, None]

                r_padding_mask = ~(data["reference_line"]["valid_mask"].any(-1))
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                bs, R, M, T, _ = trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                AV_future = (trajectory.reshape(bs, R * M, T, -1)[torch.arange(bs), flattened_probability.argmax(-1)])[:,None,:,:2] #(bs, 1, 80, 2)

                dist_prediction_cal_from_pred = pred_AV_centric-AV_future
                dist_prediction_cal_from_pred_Dist = torch.norm(dist_prediction_cal_from_pred, dim=-1)
                x_diff = (dist_prediction_cal_from_pred[..., 0:1]) #.unsqueeze(-1).type(torch.float32)  # [1296, 48, 1]
                y_diff = (dist_prediction_cal_from_pred[..., 1:2]) #.unsqueeze(-1).type(torch.float32)  # [1296, 48, 1]
                rot_prediction_cal_from_pred = torch.atan2(y_diff.squeeze(-1), x_diff.squeeze(-1)) #.unsqueeze(-1).type(torch.float32)  # [1296, 48, 1]
                out["dist_prediction_cal_from_pred"] = torch.stack((dist_prediction_cal_from_pred_Dist, rot_prediction_cal_from_pred), dim=-1)

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])
            if self.moe_CIL:
                out["hidden_q"] = self.hidden_proj_q(q)
                out["hidden_tgt_route"] = self.hidden_proj_tgt_route(tgt_route)

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            if self.no_prediction_for_CLSR:
                output_prediction = torch.zeros(
                    bs, A-1, self.future_steps, 5
                )
            else:
                output_prediction = torch.cat(
                    [
                        prediction[..., :2] + agent_pos[:, 1:A, None],
                        torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                        + agent_heading[:, 1:A, None, None],
                        prediction[..., 4:6],
                    ],
                    dim=-1,
                )
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1) # shape : bs, R
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6) # shape : bs, R, M

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2]) # shape bs, R, M, T
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory # shape : bs, R, M, T, 3(x,y,yaw)
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )

        if self.training:
            return out
        else: #direct action return for simulation
            candidate_trajectories = (
                out["candidate_trajectories"][0].cpu().numpy().astype(np.float64)
            )
            probability = out["probability"][0].cpu().numpy()

            ref_free_trajectory = (
                (out["output_ref_free_trajectory"][0].cpu().numpy().astype(np.float64))
                if "output_ref_free_trajectory" in out
                else None
            )

            if candidate_trajectories.shape[0] == 0:
                return ref_free_trajectory[:, 0, :]
            else:
                candidate_trajectories, learning_based_score = self._trim_candidates(
                    candidate_trajectories,
                    probability,
                    agent_pos,
                    agent_heading,
                    ref_free_trajectory,
                )
                best_candidate_idx = learning_based_score.argmax()
                trajectory = candidate_trajectories[best_candidate_idx, 1:]
                return trajectory[:, 0, :]