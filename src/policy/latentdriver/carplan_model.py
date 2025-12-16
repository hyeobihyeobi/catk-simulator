import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from src.policy.commons.optimzier import build_optimizer
from src.policy.commons.scheduler import build_scheduler
from .loss.GMMloss import GMMloss, get_nearest_mode_idxs
from ..commons.enc import build_model as build_enc
from .world import build_model as build_world
from .world.latent_world_model import sample_from_distribution
from .loss.KLloss import KLLoss
from .transformers.mpa_decoder import TransformerDecoderLayer
from .transformers.utils import TrainableQueryProvider
import math
from pytorch_lightning.utilities import rank_zero_info
from .modules.planning_decoder_deepseek_v2 import PlanningDecoder as deepseek_decoder
from .modules.planning_decoder_origin import PlanningDecoder_origin
from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor, DistPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder

from .layers.mlp_layer import MLPLayer

import torch.nn.functional as F

from debug_visualisation import DebugVisualisation

def unpack_action(action, B, T):
    # action: B*T, K, 7
    prob, out_model, yaw = action[...,0:1], action[...,1:6], action[...,6:]
    bs_slice = torch.arange(B * T)
    mode = prob.reshape(prob.shape[0],-1).argmax(dim=-1)
    action_ = out_model[bs_slice,mode[bs_slice],:2]
    action_ = torch.cat([action_,yaw[bs_slice,mode[bs_slice],...]],dim=-1)
    return prob, out_model, yaw, action_

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """
    Generate sine embeddings for positions.

    Parameters:
    pos_tensor (torch.Tensor): A tensor containing positions with shape (n_query, bs, 2) where the last dimension contains x, y.
    d_model (int): The dimensionality of the output embeddings (default is 256).

    Returns:
    torch.Tensor: A tensor containing the sine embeddings with shape (n_query, bs, d_model).
    """
    assert pos_tensor.size(-1) in [2, 4]
    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def apply_cross_attention(kv_feature, kv_mask, kv_pos, query_content, query_embed, attention_layer,
                            dynamic_query_center=None, layer_idx=0, use_local_attn=False, query_index_pair=None,
                            query_content_pre_mlp=None, query_embed_pre_mlp=None):
        """
        Args:
            kv_feature (B, N, C):
            kv_mask (B, N):
            kv_pos (B, N, 3):
            query_tgt (M, B, C):
            query_embed (M, B, C):
            dynamic_query_center (M, B, 2): . Defaults to None.
            attention_layer (layer):

            query_index_pair (B, M, K)

        Returns:
            attended_features: (B, M, C)
            attn_weights:
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        num_q, batch_size, d_model = query_content.shape
        searching_query = dynamic_query_center
        # searching_query = gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)
        kv_pos = kv_pos.permute(1, 0, 2)[:, :, 0:2]
        kv_pos_embed = gen_sineembed_for_position(kv_pos, hidden_dim=d_model)


        query_feature = attention_layer(
            tgt=query_content,
            query_pos=query_embed,
            query_sine_embed=searching_query,
            memory=kv_feature.permute(1, 0, 2),
            memory_key_padding_mask=~kv_mask,
            pos=kv_pos_embed,
            is_first=(layer_idx == 0)
        )  # (M, B, C)

        return query_feature

class GMMHead(nn.Module):
    def __init__(
        self,
        d_model,
        **kwargs
    ):
        super().__init__()
        self.prob_predictor = nn.Linear(d_model, 1)
        self.output_model = nn.Linear(d_model, 5)
        # # yaw is depended on xy
        self.out_yaw = nn.Linear(d_model, 1)
    def forward(self, x):
        '''
            return torch[prob, out_model, yaw]
        '''
        # prob: bs, mode, 1
        prob = self.prob_predictor(x)
        # model: bs, mode, 5
        out_model = self.output_model(x)
        # yaw bs, mode, 1
        yaw = self.out_yaw(x)
        return torch.concat([prob, out_model, yaw], dim=-1)

class MPA_blocks(nn.Module):
    def __init__(self,
                hidden_size,
                num_cross_attention_heads,
                 **kwargs):
        super().__init__()
        self.decoder_layer = TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_cross_attention_heads, dim_feedforward=hidden_size * 4, dropout=0.1, normalize_before=False, keep_query_pos=False,
            rm_self_attn_decoder=False, use_local_attn=False
        )
        self.gmmhead = GMMHead(hidden_size)
    def forward(self, latent, q_content, q_emb, layer_idx, action_embedding):
        B,T,M,_ = latent.shape
        latent = latent.view(B*T,M,-1)
        query_content = apply_cross_attention(
                kv_feature=latent,
                kv_mask=torch.ones((B*T,M), device=latent.device, dtype=torch.bool),
                kv_pos=torch.zeros((B*T,M,3),device=latent.device, dtype=torch.float32),
                query_content=q_content,
                query_embed=q_emb, #only vaild for first layer
                attention_layer=self.decoder_layer,
                dynamic_query_center = action_embedding.permute(1,0,2),
                layer_idx=layer_idx,
            )
        action_dis = self.gmmhead(query_content.permute(1,0,2))
        return action_dis, query_content

class CarPLAN(pl.LightningModule):
    def __init__(
        self,
        max_length=None,
        eval_context_length=None,
        pretrain_enc = None,
        freeze_enc = False,
        encoder = None,
        pretrain_world = None,
        freeze_world = False,
        world = None,
        function = 'enc-LWM-MPP',
        mode = 6,
        num_of_decoder = 3,
        num_cross_attention_heads=4,
        est_layer = 0,
        bert_chunk_size = None,

        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=11,
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
        shared_expert_weight = 1.0,
        is_carplan = False,

        **kwargs,
    ):
        super().__init__()
        self.est_layer = est_layer
        self.function = function
        self.act_dim = world.act_dim
        self.max_length = max_length
        self.hidden_size = world.hidden_size
        self.eval_context_length = eval_context_length
        self.ordering = world.ordering
        if bert_chunk_size is not None and bert_chunk_size <= 0:
            bert_chunk_size = None
        self.initial_bert_chunk_size = bert_chunk_size
        self.runtime_bert_chunk_size = bert_chunk_size
        self.init_enc(encoder, pretrain_enc, freeze_enc)
        assert self.function in ['enc-LWM-MPP', 'enc-MPP']
        # if self.function == 'enc-LWM-MPP':
        #     print("Initating world")
        #     self.init_world(world,pretrain_world, freeze_world)
        # MPAD
        # hidden_size = world.hidden_size
        # self.action_prob_emb = nn.Linear(7, hidden_size)
        # self.query_pe = TrainableQueryProvider(num_queries=mode, num_query_channels=hidden_size, init_scale=0.01)
        # self.action_distribution_queries = TrainableQueryProvider(num_queries=mode, num_query_channels=hidden_size, init_scale=0.01)
        # self.mpad_blocks = nn.ModuleList([MPA_blocks(hidden_size=hidden_size, num_cross_attention_heads=num_cross_attention_heads) for _ in range(num_of_decoder)])

        #         self.imagine_query = None
        self.av_cat = av_cat
        self.no_displacement_for_CLSR = no_displacement_for_CLSR
        self.no_prediction_for_CLSR = no_prediction_for_CLSR
        self.history_steps = history_steps
        
        self.is_carplan = is_carplan

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
            use_lane_boundary=False,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        if is_carplan:
            self.displacement_encoder_blocks = nn.ModuleList(
                TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
                for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
            )
            self.displacement_norm = nn.LayerNorm(dim)
            self.dist_predictor = DistPredictor(dim=dim, future_steps=future_steps)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)

        if self.av_cat:
            self.av_cat_x_proj = nn.Linear(2 * dim, dim)

        if is_carplan:
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
        else:
            self.planning_decoder = PlanningDecoder_origin(
                    num_mode=num_modes,
                    decoder_depth=decoder_depth,
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    dropout=dropout,
                    cat_x=cat_x,
                    future_steps=future_steps,
                )

        # if self.ref_free_traj:
        self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.loc_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)
        self.yaw_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)
        self.vel_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)

        self.optim_conf = kwargs['optimizer']
        self.sched_conf = kwargs['scheduler']
        self.lr = kwargs['learning_rate']
        
        self.tl_types = {0: 3, 
                         1: 3, 
                         2: 3, 
                         3: 3, 
                         4: 2,
                         5: 1,
                         6: 0,
                         7: 3,
                         8: 3}

    def init_world(self, world_conf, pretrain_world, freeze_world):
        self.world_model = build_world(world_conf)
        if pretrain_world is not None:
            self.world_model.load_state_dict(torch.load(pretrain_world))
            print(f'Loaded pretained world from {pretrain_world}')
        if freeze_world:
            for para in self.world_model.parameters():
                para.requires_grad = False

    def init_enc(self, bert_conf, pretrain_enc, freeze_enc):
        self.bert = build_enc(bert_conf)
        if pretrain_enc is not None:
            self.bert.load_state_dict(torch.load(pretrain_enc))
            print(f'Loaded pretained enc from {pretrain_enc}')
        if freeze_enc:
            for para in self.bert.parameters():
                para.requires_grad = False

    def _encode_with_bert(self, flattened_states):
        total = flattened_states.size(0)
        target_chunk = self.runtime_bert_chunk_size or total
        chunk_size = min(target_chunk, total)
        while True:
            try:
                if chunk_size >= total:
                    outputs = self.bert(flattened_states, return_full_length=True)
                else:
                    chunks = []
                    for start in range(0, total, chunk_size):
                        end = start + chunk_size
                        chunks.append(self.bert(flattened_states[start:end], return_full_length=True))
                    outputs = torch.cat(chunks, dim=0)
                if self.runtime_bert_chunk_size != chunk_size:
                    self.runtime_bert_chunk_size = chunk_size
                    if chunk_size != target_chunk:
                        rank_zero_info(f"[LantentDriver] Adjusted BERT chunk size to {chunk_size} for available GPU memory.")
                return outputs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk_size <= 1:
                    raise
                new_chunk = max(1, chunk_size // 2)
                if new_chunk == chunk_size:
                    new_chunk = chunk_size - 1
                rank_zero_info(f"[LantentDriver] BERT chunk size {chunk_size} caused OOM. Retrying with {new_chunk}.")
                chunk_size = max(1, new_chunk)

    def forward(
        self,
        ss,
        reference_lines=None,
    ):
        data = self.ss_to_datadict(ss)

        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1) #torch.Size([B, N+1+M, 2])
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1) #torch.Size([B, N+1+M])
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1) #torch.Size([B, N+1+M, 3])

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        
        M_shape = x_polygon.shape[1]

        x = torch.cat([x_agent, x_polygon], dim=1)

        pos_embed = self.pos_emb(pos)
        x = x + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x) #torch.Size([B, N+1+M+S, 128])

        if self.no_prediction_for_CLSR:
            prediction = torch.zeros(bs, A-1, self.future_steps, 6).to(x)
        else:
            prediction = self.agent_predictor(x[:, :A]) #torch.Size([B, N, Future_step, 6])

        x_scene_encoder = None

        if self.is_carplan:
            for dl in self.displacement_encoder_blocks:
                x = dl(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
            x = self.displacement_norm(x)

            if self.av_cat:
                cat_av_dist = torch.cat([x[:, 1:], x[:, 0:1].repeat(1, A+M_shape-1, 1)], dim=-1)
                cat_av_dist = self.av_cat_x_proj(cat_av_dist)
                dist_prediction = self.dist_predictor(cat_av_dist)
            else:
                dist_prediction = self.dist_predictor(x[:, 1:])
        else:
            dist_prediction = None

        # reference_lines[]
        r_padding_mask = ~(reference_lines["valid_mask"][:bs].any(-1))  # (bs, R)
        ref_line_available = ~(r_padding_mask.all(-1).all())
        # ref_line_available = reference_lines["position"].shape[1] > 0
        R, M = reference_lines["position"].shape[1], 12

        if ref_line_available:
            trajectory, probability, pred_scenario_type, q, tgt_route, gates, load, gates_dict, dist_predictions, scores_list = self.planning_decoder(
                reference_lines, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask, "enc_pos": pos, "x_scene_encoder": None}
            )
        else:
            trajectory, probability, pred_scenario_type, q, tgt_route, gates, load, gates_dict, scores_list = None, None, None, None, None, None, None, None, None

        # if self.ref_free_traj:
        # ref_free_traj = self.ref_free_decoder(x[:, 1, :]).reshape(
        #     bs, 80, 4
        # )
        ref_free_traj = self.ref_free_decoder(x[torch.arange(bs), data['is_sdc_index']]).reshape(
                bs, 80, 4
            )

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
            "load": load,
            "scores_list": scores_list,
            "ref_free_trajectory": ref_free_traj,
        }

        out["dist_prediction"] = dist_prediction

        latent_dist = None
        rep_dist = None

        if not self.training:
            ref_free_traj_angle = torch.arctan2(
                ref_free_traj[..., 3], ref_free_traj[..., 2]
            )
            ref_free_traj = torch.cat(
                [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
            )
            output_ref_free_trajectory = ref_free_traj

            if trajectory is not None:
                r_padding_mask = ~reference_lines["valid_mask"].any(-1) # shape : bs, R
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6) # shape : bs, R, M

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2]) # shape bs, R, M, T
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ][:, 0]
                best_trajectory[torch.where(reference_lines["valid_mask"].any(-1).sum(-1) == 0)[0]] = \
                    output_ref_free_trajectory[torch.where(reference_lines["valid_mask"].any(-1).sum(-1) == 0)[0], 0]
            else:
                best_trajectory = output_ref_free_trajectory[:, 0]
        else:
            best_trajectory = None
            output_ref_free_trajectory = None

#         return out, best_trajectory,latent_dist, rep_dist
        return out, output_ref_free_trajectory,latent_dist, rep_dist

    def calculate_additional_ego_states(
        self, cur_velocity, angle_diff, wheel_base, dt=0.1
    ):
        # cur_velocity = current_state.dynamic_car_state.rear_axle_velocity_2d.x
        # angle_diff = current_state.rear_axle.heading - prev_state.rear_axle.heading
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = angle_diff / dt
        
        eps = 0.2
        valid_mask = cur_velocity.abs() >= eps  
        safe_velocity = cur_velocity.abs().clamp(min=eps)
        steering_angle = torch.atan(yaw_rate * wheel_base.unsqueeze(-1) / safe_velocity)
        steering_angle = steering_angle.clamp(-2/3 * math.pi, 2/3 * math.pi)
        yaw_rate = yaw_rate.clamp(-0.95, 0.95)
        
        steering_angle = torch.where(valid_mask, steering_angle, torch.zeros_like(steering_angle))
        yaw_rate = torch.where(valid_mask, yaw_rate, torch.zeros_like(yaw_rate))

        return steering_angle, yaw_rate
        
    def ss_to_datadict(self, ss):
        if not isinstance(ss, dict):
            raise TypeError("ss_to_datadict expects a dict with separated attributes (his_veh_trajs, ...).")

        def _to_torch(x):
            return x if torch.is_tensor(x) else torch.tensor(x, device=self.device)

        his = _to_torch(ss["his_veh_trajs"])
        # Flatten leading device/batch axes into shape (envs, agents, T, feat)
        if his.dim() == 5:
            his = his.reshape(-1, his.shape[2], his.shape[3], his.shape[4])

        # Zero history timesteps that fall outside the [-50, 50] ROI in x/y (ROI size [50, 50] * 2).
        if his.shape[-1] >= 3:
            roi_half_extent = his.new_tensor([50.0, 50.0])
            pos = his[..., 1:3]
            within_roi = (pos >= -roi_half_extent) & (pos <= roi_half_extent)
            within_roi = within_roi.all(dim=-1)  # shape: (envs, agents, T)
            his = torch.where(within_roi.unsqueeze(-1), his, torch.zeros_like(his))

        # Mask out invalid history entries using the provided valid flag (feat index 10).
        if his.shape[-1] > 10:
            hist_valid_mask = (his[..., 10] == 1).unsqueeze(-1)
            his = his * hist_valid_mask

        current_state_tmp = _to_torch(ss["vehicle_segments"])
        # Flatten leading device/batch axes; keep agent/time/feat structure if present.
        if current_state_tmp.dim() == 5:
            # (num_devices, batch, agents, T, feat) -> (envs, agents, T, feat)
            current_state_tmp = current_state_tmp.reshape(-1, current_state_tmp.shape[2], current_state_tmp.shape[3], current_state_tmp.shape[4])
        elif current_state_tmp.dim() == 4:
            # Could be (envs, agents, T, feat) or (num_devices, batch, agents, feat)
            # Heuristic: if last dim is feature size (< time length), and third dim matches his time len, treat third dim as time.
            if his.dim() == 4 and current_state_tmp.shape[2] == his.shape[2]:
                pass  # already (envs, agents, T, feat)
            else:
                current_state_tmp = current_state_tmp.reshape(-1, current_state_tmp.shape[2], current_state_tmp.shape[3])
        elif current_state_tmp.dim() == 3:
            # Assume (envs, agents, feat)
            pass
        elif current_state_tmp.dim() == 2:
            current_state_tmp = current_state_tmp.unsqueeze(1)  # (envs, 1, feat)

        # If time axis exists, take the latest timestep.
        if current_state_tmp.dim() == 4:
            current_state_tmp = current_state_tmp[..., -1, :]  # (envs, agents, feat)

        ego_state = current_state_tmp[(current_state_tmp[..., 0] == 4)]
        # ego_state = current_state_tmp[:, 0, :]  # assume ego is agent index 0
        xy = ego_state[..., 1:3]
        heading = ego_state[..., 5:6]
        v_x = ego_state[..., 6:7] #torch.sqrt(ego_state[..., 6:7] ** 2 + ego_state[..., 7:8] ** 2)
        a_x = ego_state[..., 8:9] #torch.sqrt(ego_state[..., 8:9] ** 2 + ego_state[..., 9:10] ** 2)
        ego_wheelbase = 0.6 * ego_state[:, 4]
        angle_diff = heading - his[(current_state_tmp[..., 0] == 4)][:, 9, 5:6]
        steering_angle, yaw_rate = self.calculate_additional_ego_states(v_x, angle_diff, ego_wheelbase)
        current_state = torch.cat(
            [xy, heading, v_x, a_x, steering_angle, yaw_rate],
            dim=-1,
        )


        road_obs_arr = _to_torch(ss['roadgraph_obs'])
        map_valid_mask = _to_torch(ss['valid_mask'])
        sample_points = 21
        ids_arr = road_obs_arr[..., 4].long() #roadgraph 고유 아이디
        valid_mask = (road_obs_arr.sum(dim=-1) != 0)
        sampled_paths: list[list[torch.Tensor]] = []
        sampled_types: list[list[torch.Tensor]] = []
        sampled_ids: list[list[torch.Tensor]] = []
        sampled_on_route: list[list[torch.Tensor]] = []
        sampled_tl_status: list[list[torch.Tensor]] = []
        sampled_speed_limit: list[list[torch.Tensor]] = []
        sampled_has_speed_limit: list[list[torch.Tensor]] = []
        max_paths = 0

        point_on_route_raw = _to_torch(ss.get("point_on_route", torch.zeros_like(ids_arr, dtype=torch.bool)))
        point_tl_status_raw = _to_torch(ss.get("point_tl_status", torch.zeros_like(ids_arr)))
        point_speed_limit_raw = _to_torch(ss.get("point_speed_limit", torch.zeros_like(ids_arr)))
        point_has_speed_limit_raw = _to_torch(ss.get("point_has_speed_limit", torch.zeros_like(ids_arr, dtype=torch.bool)))

        def _reshape_points(arr, dtype=None):
            if dtype is not None and arr.dtype != dtype:
                arr = arr.to(dtype)
            if arr.dim() > 2:
                return arr.reshape(road_obs_arr.shape[0], road_obs_arr.shape[1])
            return arr

        point_on_route = _reshape_points(point_on_route_raw, dtype=torch.bool)
        point_tl_status = _reshape_points(point_tl_status_raw)
        point_speed_limit = _reshape_points(point_speed_limit_raw)
        point_has_speed_limit = _reshape_points(point_has_speed_limit_raw, dtype=torch.bool)

        for b in range(road_obs_arr.shape[0]):
    #         valid_points = road_obs_arr[b][jnp.where(valid_mask[b])[0]]
    #         valid_ids = ids_arr[b][valid_mask[b]].astype(jnp.int32)
            valid_points = torch.where(
                valid_mask[b].unsqueeze(-1),
                road_obs_arr[b],
                torch.zeros_like(road_obs_arr[b]),
            )
            valid_ids = torch.where(
                valid_mask[b],
                ids_arr[b],
                torch.full_like(ids_arr[b], -1),
            )
            valid_on_route = torch.where(
                valid_mask[b],
                point_on_route[b] if point_on_route.numel() else torch.zeros_like(valid_ids, dtype=torch.bool),
                torch.zeros_like(valid_ids, dtype=torch.bool),
            )
            valid_tl_status = torch.where(
                valid_mask[b],
                point_tl_status[b] if point_tl_status.numel() else torch.zeros_like(valid_ids),
                torch.zeros_like(valid_ids),
            )
            valid_speed_limit = torch.where(
                valid_mask[b],
                point_speed_limit[b] if point_speed_limit.numel() else torch.zeros_like(valid_ids, dtype=road_obs_arr.dtype),
                torch.zeros_like(valid_ids, dtype=road_obs_arr.dtype),
            )
            valid_has_speed_limit = torch.zeros_like(valid_ids, dtype=torch.bool)
            batch_paths = []
            batch_types = []
            batch_ids = []
            batch_on_route = []
            batch_tl_status = []
            batch_speed_limit = []
            batch_has_speed_limit = []
            for uid in torch.unique(valid_ids[valid_ids >= 0]):
                pts = valid_points[valid_ids == uid]
                if pts.shape[0] < 2:
                    continue
                uid_mask = valid_ids == uid
                path_xyz = torch.cat(
                    [
                        pts[:, :2],
                        torch.zeros((pts.shape[0], 1), device=pts.device, dtype=pts.dtype),
                    ],
                    dim=-1,
                )
                sampled = self.interpolate_polyline(path_xyz, sample_points)
                batch_paths.append(sampled)
                # roadgraph_obs columns: [x, y, yaw, type, id]; use column 3 for type
                type_val = pts[0, 3]
                type_val = torch.where(
                    type_val == 18,
                    torch.tensor(1, device=pts.device, dtype=type_val.dtype),
                    0,
                )
                batch_types.append(torch.full((sample_points,), type_val, device=pts.device, dtype=sampled.dtype))
                batch_ids.append(torch.full((sample_points,), uid, device=pts.device, dtype=torch.long))
                # Aggregate scalar attributes per polyline.
                on_route_val = valid_on_route[uid_mask].any() if valid_on_route.numel() else torch.zeros((), device=pts.device, dtype=torch.bool)
                tl_status_val = torch.tensor(self.tl_types[int(valid_tl_status[uid_mask].max())]).to(on_route_val) if valid_tl_status.numel() else torch.zeros((), device=pts.device, dtype=valid_tl_status.dtype)
                speed_limit_val = valid_speed_limit[uid_mask].max() if valid_speed_limit.numel() else torch.zeros((), device=pts.device, dtype=road_obs_arr.dtype)
                has_speed_limit_val = valid_has_speed_limit[uid_mask].any() if valid_has_speed_limit.numel() else torch.zeros((), device=pts.device, dtype=torch.bool)
                batch_on_route.append(on_route_val)
                batch_tl_status.append(tl_status_val)
                batch_speed_limit.append(speed_limit_val)
                batch_has_speed_limit.append(has_speed_limit_val)
            max_paths = max(max_paths, len(batch_paths))
            sampled_paths.append(batch_paths)
            sampled_types.append(batch_types)
            sampled_ids.append(batch_ids)
            sampled_on_route.append(batch_on_route)
            sampled_tl_status.append(batch_tl_status)
            sampled_speed_limit.append(batch_speed_limit)
            sampled_has_speed_limit.append(batch_has_speed_limit)
        padded_paths = []
        padded_types = []
        padded_ids = []
        padded_on_route = []
        padded_tl_status = []
        padded_speed_limit = []
        padded_has_speed_limit = []
        for paths, types, ids_list in zip(sampled_paths, sampled_types, sampled_ids):
            if len(paths) < max_paths:
                paths = paths + [torch.zeros((sample_points, 3), device=road_obs_arr.device, dtype=road_obs_arr.dtype)] * (max_paths - len(paths))
                types = types + [torch.zeros((sample_points,), device=road_obs_arr.device, dtype=road_obs_arr.dtype)] * (max_paths - len(types))
                ids_list = ids_list + [torch.zeros((sample_points,), device=road_obs_arr.device, dtype=torch.long)] * (max_paths - len(ids_list))
            padded_paths.append(torch.stack(paths, dim=0) if paths else torch.zeros((0, sample_points, 3), device=road_obs_arr.device, dtype=road_obs_arr.dtype))
            padded_types.append(torch.stack(types, dim=0) if types else torch.zeros((0, sample_points), device=road_obs_arr.device, dtype=road_obs_arr.dtype))
            padded_ids.append(torch.stack(ids_list, dim=0) if ids_list else torch.zeros((0, sample_points), device=road_obs_arr.device, dtype=torch.long))
        for on_route_list, tl_list, spd_list, has_spd_list in zip(sampled_on_route, sampled_tl_status, sampled_speed_limit, sampled_has_speed_limit):
            if len(on_route_list) < max_paths:
                pad_len = max_paths - len(on_route_list)
                on_route_list = on_route_list + [torch.zeros((), device=road_obs_arr.device, dtype=torch.bool)] * pad_len
                tl_list = tl_list + [torch.zeros((), device=road_obs_arr.device, dtype=road_obs_arr.dtype)] * pad_len
                spd_list = spd_list + [torch.zeros((), device=road_obs_arr.device, dtype=road_obs_arr.dtype)] * pad_len
                has_spd_list = has_spd_list + [torch.zeros((), device=road_obs_arr.device, dtype=torch.bool)] * pad_len
            padded_on_route.append(torch.stack(on_route_list, dim=0) if on_route_list else torch.zeros((0,), device=road_obs_arr.device, dtype=torch.bool))
            padded_tl_status.append(torch.stack(tl_list, dim=0) if tl_list else torch.zeros((0,), device=road_obs_arr.device, dtype=road_obs_arr.dtype))
            padded_speed_limit.append(torch.stack(spd_list, dim=0) if spd_list else torch.zeros((0,), device=road_obs_arr.device, dtype=road_obs_arr.dtype))
            padded_has_speed_limit.append(torch.stack(has_spd_list, dim=0) if has_spd_list else torch.zeros((0,), device=road_obs_arr.device, dtype=torch.bool))

        if len(padded_paths) > 0:
            roadgraph_sampled = torch.stack(padded_paths, dim=0)
            roadgraph_sampled_type = torch.stack(padded_types, dim=0)
            roadgraph_sampled_id = torch.stack(padded_ids, dim=0)
            polygon_on_route_vals = torch.stack(padded_on_route, dim=0)
            polygon_tl_status_vals = torch.stack(padded_tl_status, dim=0)
            polygon_speed_limit_vals = torch.stack(padded_speed_limit, dim=0)
            polygon_has_speed_limit_vals = torch.stack(padded_has_speed_limit, dim=0)
        else:
            roadgraph_sampled = torch.zeros((road_obs_arr.shape[0], 0, sample_points, 3), device=road_obs_arr.device, dtype=road_obs_arr.dtype)
            roadgraph_sampled_type = torch.zeros((road_obs_arr.shape[0], 0, sample_points), device=road_obs_arr.device, dtype=road_obs_arr.dtype)
            roadgraph_sampled_id = torch.zeros((road_obs_arr.shape[0], 0, sample_points), device=road_obs_arr.device, dtype=torch.long)
            polygon_on_route_vals = torch.zeros((road_obs_arr.shape[0], 0), device=road_obs_arr.device, dtype=torch.bool)
            polygon_tl_status_vals = torch.zeros((road_obs_arr.shape[0], 0), device=road_obs_arr.device, dtype=road_obs_arr.dtype)
            polygon_speed_limit_vals = torch.zeros((road_obs_arr.shape[0], 0), device=road_obs_arr.device, dtype=road_obs_arr.dtype)
            polygon_has_speed_limit_vals = torch.zeros((road_obs_arr.shape[0], 0), device=road_obs_arr.device, dtype=torch.bool)

        B, M, P = roadgraph_sampled.shape[0], roadgraph_sampled.shape[1], sample_points
        # point_position = roadgraph_sampled[..., :2] #.unsqueeze(2)  # (B, M, 1, P, 2)
        # forward difference with zero padding on the last point
        point_vector = (roadgraph_sampled[..., 1:, :2] - roadgraph_sampled[..., :-1, :2])
        point_position = roadgraph_sampled[:, :, :-1, :2]
        # zero_tail = torch.zeros_like(diff[..., :1, :])
        # point_vector = torch.cat([diff, zero_tail], dim=-2).unsqueeze(2)  # (B, M, 1, P, 2)
        point_side = torch.zeros((B, M, 1), device=road_obs_arr.device, dtype=torch.int8)
        point_orientation = torch.arctan2(point_vector[..., 1], point_vector[..., 0]) #roadgraph_sampled[..., 2].unsqueeze(2)  # (B, M, 1, P)
        
        polygon_center = torch.cat((roadgraph_sampled[:, :, int((sample_points-1) / 2), :2], point_orientation[:, :, int((sample_points-1) / 2)].unsqueeze(-1)), dim=-1)
        polygon_position = roadgraph_sampled[..., 0, :2]
        polygon_orientation = point_orientation[..., 0]
        
        polygon_type = roadgraph_sampled_type[..., 0] if roadgraph_sampled_type.numel() else torch.zeros((B, M), device=road_obs_arr.device, dtype=road_obs_arr.dtype)
        polygon_on_route = polygon_on_route_vals
        polygon_tl_status = polygon_tl_status_vals
        polygon_speed_limit = polygon_speed_limit_vals
        polygon_has_speed_limit = polygon_has_speed_limit_vals
        polygon_road_block_id = roadgraph_sampled_id[..., 0] if roadgraph_sampled_id.numel() else torch.zeros((B, M), device=road_obs_arr.device, dtype=torch.long)
        valid_mask = point_position.abs().sum(dim=-1) != 0 #torch.any(roadgraph_sampled.abs().sum(dim=-1) != 0, dim=-1)
        # Align map_valid_mask to the sampled roadgraph shape (B, M, P) so downstream code can use the same masking.
        # map_valid_mask_raw = ss.get("map_valid_mask", None)
        # if map_valid_mask_raw is not None:
        #     map_valid_mask_raw = _to_torch(map_valid_mask_raw)
        #     # Flatten any leading device/batch dims and trim/pad to match sample_points per segment.
        #     map_valid_mask_raw = map_valid_mask_raw.reshape(B, -1)
        #     if map_valid_mask_raw.shape[-1] >= sample_points:
        #         map_valid_mask_sampled = map_valid_mask_raw[:, : sample_points]
        #     else:
        #         pad_len = sample_points - map_valid_mask_raw.shape[-1]
        #         map_valid_mask_sampled = torch.cat(
        #             [map_valid_mask_raw, torch.zeros((B, pad_len), device=map_valid_mask_raw.device, dtype=map_valid_mask_raw.dtype)],
        #             dim=-1,
        #         )
        #     # Broadcast to (B, M, P) following roadgraph_sampled.
        #     map_valid_mask = map_valid_mask_sampled[:, None, :].expand(B, M, sample_points)
        # else:
        #     map_valid_mask = torch.ones((B, M, sample_points), device=road_obs_arr.device, dtype=torch.bool)

        # his shape now: (num_envs, num_agents, T_hist, feat)
        agent_pos = his[..., 1:3]                     # (B, A, T, 2)
        agent_shape = his[..., 3:5]                   # (B, A, T, 2) -> width, length
        agent_heading = torch.deg2rad(his[..., 5])    # yaw(deg) -> rad
        agent_vel = his[..., 6:8]                     # vx, vy
        agent_acc = his[..., 8:10]                    # ax, ay
        agent_valid = (his[..., 10] == 1).bool()      # (B, A, T)
        # agent_category = his[..., 0, 0]               # type id (2=vehicle, 4=SDC set upstream)
        agent_category = torch.where(his[..., 0, 0] == 2, torch.tensor(1, device=his.device), his[..., 0, 0])
        agent_category[torch.arange(B), torch.where(current_state_tmp[..., 0] == 4)[1]] = 0

#         DebugVisualisation().plot_sampled_map_and_route(
#             point_position,
#             agent_pos,
#             ids=polygon_type,
#             batch_idx=0,
#         )


#         # Prepare lightweight CPU numpy inputs for visualization to match plot_map_jax expectations.
#         map_for_vis = _to_torch(ss['roadgraph_obs'][..., :2])
#         ids_for_vis = _to_torch(ss['roadgraph_obs'][..., 4])
#         types_for_vis = _to_torch(ss['roadgraph_obs'][..., 3:4])
# 
#         # Flatten all non-batch dimensions into a single point axis so each x/y is scalar,
#         # then add a dummy lane dimension.
#         map_for_vis = map_for_vis.reshape(map_for_vis.shape[0], -1, 2)   # (B, N, 2)
#         ids_for_vis = ids_for_vis.reshape(ids_for_vis.shape[0], -1)      # (B, N)
#         types_for_vis = types_for_vis.reshape(types_for_vis.shape[0], -1, types_for_vis.shape[-1])  # (B, N, 1)
# 
#         map_for_vis = map_for_vis[:1].unsqueeze(1)    # (1, 1, N, 2)
#         ids_for_vis = ids_for_vis[:1].unsqueeze(1)    # (1, 1, N)
#         types_for_vis = types_for_vis[:1].unsqueeze(1)  # (1, 1, N, 1)
# 
#         map_for_vis = map_for_vis.detach().cpu().numpy()
#         ids_for_vis = ids_for_vis.detach().cpu().numpy()
#         types_for_vis = types_for_vis.detach().cpu().numpy()
# 
#         DebugVisualisation().plot_map_jax(
#             map_for_vis,
#             None,
#             ids=ids_for_vis,
#             types=types_for_vis,
#             batch_idx=0,
#         )

#         point_position = _to_torch(ss['point_position'])
#         point_vector = _to_torch(ss['point_vector'])
#         point_side = _to_torch(ss['point_side'])
#         point_orientation = _to_torch(ss['point_orientation'])
#         polygon_center = _to_torch(ss['polygon_center'])
#         polygon_position = _to_torch(ss['polygon_position'])
#         polygon_orientation = _to_torch(ss['polygon_orientation'])
#         polygon_type = _to_torch(ss['polygon_type'])
#         polygon_on_route = _to_torch(ss['polygon_on_route'])
#         polygon_tl_status = _to_torch(ss['polygon_tl_status'])
#         polygon_speed_limit = _to_torch(ss['polygon_speed_limit'])
#         polygon_has_speed_limit = _to_torch(ss['polygon_has_speed_limit'])
#         polygon_road_block_id = _to_torch(ss['polygon_road_block_id'])
#         valid_mask = _to_torch(ss['valid_mask'])

        data_dict = {
            "current_state": current_state,
            "is_sdc_index": torch.where(current_state_tmp[..., 0] == 4)[1],
            "agent": {
                "position": agent_pos,
                "heading": agent_heading,
                "velocity": agent_vel,
                "acceleration": agent_acc,
                "shape": agent_shape,
                "category": agent_category,
                "valid_mask": agent_valid,
            } ,
            "map": {
                "point_position": point_position.unsqueeze(2),
                "point_vector": point_vector.unsqueeze(2),
                "point_side": point_side,
                "point_orientation": point_orientation.unsqueeze(2),
                "polygon_center": polygon_center,
                "polygon_position": polygon_position,
                "polygon_orientation": polygon_orientation,
                "polygon_type": polygon_type,
                "polygon_on_route": polygon_on_route,
                "polygon_tl_status": polygon_tl_status,
                "polygon_speed_limit": polygon_speed_limit,
                "polygon_has_speed_limit": polygon_has_speed_limit.bool(),
                "polygon_road_block_id": polygon_road_block_id,
                "valid_mask": valid_mask,
            }
        }

        return data_dict

    def get_predictions(
        self, states, actions, reference_lines, timesteps, num_envs=1, **kwargs
    ):
        # states는 dict 형태의 관측(분리된 attribute). 텐서가 아니면 텐서로 변환.
        if isinstance(states, dict):
            states = {
                k: (v if torch.is_tensor(v) else torch.tensor(v, device=self.device))
                for k, v in states.items()
            }
        else:
            raise TypeError("get_predictions expects states as a dict of tensors from preprocess_data_dist_jnp.")

        padding_mask = None

        for bs in range(num_envs):
            if reference_lines is not None:
                for ref_idx, ref_lines in enumerate(reference_lines["position"][bs]):
                    ref_valid_mask = reference_lines["valid_mask"][bs, ref_idx]
                    ref_lines_valid = ref_lines[ref_valid_mask]
                    diffs = ref_lines_valid[1:] - ref_lines_valid[:-1]           # (N-1, 2)
                    dists = torch.norm(diffs, dim=1)             # 각 구간 거리
                    total_length = dists.sum()
                    if total_length > 120:
                        reference_lines["valid_mask"][bs, ref_idx] = False
            else:
                reference_lines = {
                    "position": torch.zeros((1, 0, 80, 2), device=self.device),
                    "vector": torch.zeros((1, 0, 80, 2), device=self.device),
                    "orientation": torch.zeros((1, 0, 80), device=self.device),
                    "valid_mask": torch.zeros((1, 0), dtype=torch.bool, device=self.device),
                    "future_projection": torch.zeros((1, 0, 3), device=self.device),
                }

        _, action, _, _ = self.forward(
            states,
            actions,
            reference_lines,
            timesteps,
            padding_mask=padding_mask,
        )
        return action


    def interpolate_polyline(self, points, t: int):
        """copy from av2-api"""

        if not torch.is_tensor(points):
            points = torch.as_tensor(points, device=self.device)
        if points.ndim != 2:
            raise ValueError("Input array must be (N,2) or (N,3) in shape.")

        # the number of points on the curve itself
        n, _ = points.shape

        # equally spaced in arclength -- the number of points that will be uniformly interpolated
        eq_spaced_points = torch.linspace(0, 1, t, device=points.device, dtype=points.dtype)

        # Compute the chordal arclength of each segment.
        # Compute differences between each x coord, to get the dx's
        # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
        chordlen: torch.Tensor = torch.norm(points[1:] - points[:-1], dim=1)
        # Normalize the arclengths to a unit total
        chordlen = chordlen / torch.clamp(chordlen.sum(), min=1e-8)
        # cumulative arclength

        cumarc: torch.Tensor = torch.zeros(len(chordlen) + 1, device=points.device, dtype=points.dtype)
        cumarc[1:] = torch.cumsum(chordlen, dim=0)

        # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
        tbins: torch.Tensor = torch.bucketize(eq_spaced_points, cumarc).to(torch.long)

        # #catch any problems at the ends
        tbins = torch.where((tbins <= 0) | (eq_spaced_points <= 0), torch.ones_like(tbins), tbins)
        tbins = torch.where((tbins >= n) | (eq_spaced_points >= 1), torch.full_like(tbins, n - 1), tbins)

        chordlen_safe = torch.where(chordlen == 0, chordlen + 1e-6, chordlen)

        s = (eq_spaced_points - cumarc[tbins - 1]) / chordlen_safe[tbins - 1]
        anchors = points[tbins - 1, :]
        # broadcast to scale each row of `points` by a different row of s
        offsets = (points[tbins, :] - points[tbins - 1, :]) * s.unsqueeze(-1)
        points_interp: torch.Tensor = anchors + offsets

        return points_interp

    def get_planning_loss(self, reference_lines, trajectory, probability, target_valid_mask, target, bs):
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        num_valid_points = target_valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~(reference_lines["valid_mask"][:bs].any(-1))  # (bs, R)
        unvalid_batch_mask = r_padding_mask.all(-1)
        future_projection = reference_lines["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / 10
        ).long()
        target_m_index.clamp_(min=0, max=12 - 1)

        target_label = torch.zeros_like(probability)
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]

        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        reg_loss = (reg_loss * target_valid_mask)[~unvalid_batch_mask].sum() / target_valid_mask[~unvalid_batch_mask].sum()

        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        cls_loss = F.cross_entropy(
            probability[~unvalid_batch_mask].reshape((bs-unvalid_batch_mask.sum()), -1), target_label[~unvalid_batch_mask].reshape((bs-unvalid_batch_mask.sum()), -1).detach()
        )

        return reg_loss, cls_loss

    def get_prediction_loss(self, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """
        valid_mask = valid_mask.bool()
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def training_step(self, batch, batch_idx):

#         (ss, position, vector, orientation, valid_mask, future_projection, target, target_vel, target_valid_mask, is_sdc) = batch
        (ss, sdc_gt, agent_gt, reference_lines) = batch
        B, _, _, _ = ss['his_veh_trajs'].shape

        import matplotlib.pyplot as plt
        # route_feat = ss[:, :, :20]
        # agent_feat = ss[:, :, 20:148]
        # road_feat = ss[:, :, 148:]

        # agent_mask = (ss[:, 1, 20:148, 0] == 2)

        # for bs in range(B):
        #     sdc_idx = torch.where(is_sdc[bs])[0]
        #     for ag_idx in range(128):
        #         gt = target[bs, ag_idx, :, :2][target_valid_mask[bs, ag_idx]].cpu() + agent_feat[bs, -1, ag_idx, 1:3].cpu()
        #         if ag_idx == sdc_idx:
        #             ss_valid_mask = (agent_feat[bs, :, ag_idx, 0] == 4)
        #             plt.plot(agent_feat[bs, :, ag_idx, 1][ss_valid_mask].cpu(), agent_feat[bs, :, ag_idx, 2][ss_valid_mask].cpu(), "bo")
        #             plt.plot(gt[:, 0], gt[:, 1], 'r-')
        #         else:
        #             ss_valid_mask = (agent_feat[bs, :, ag_idx, 0] == 2)
        #             plt.plot(agent_feat[bs, :, ag_idx, 1][ss_valid_mask].cpu(), agent_feat[bs, :, ag_idx, 2][ss_valid_mask].cpu(), "go")
        #             # if ss_valid_mask[-1]: #.all():
        #             if agent_mask[bs, ag_idx]:
        #                 plt.plot(gt[:, 0], gt[:, 1], 'g-')
        #     for rd_idx in range(40):
        #         ss_valid_mask = (road_feat[bs, :, rd_idx, 0] == 3)
        #         plt.plot(road_feat[bs, :, rd_idx, 1][ss_valid_mask].cpu(), road_feat[bs, :, rd_idx, 2][ss_valid_mask].cpu(), "ko")
        #     for rt_idx in range(20):
        #         ss_valid_mask = (route_feat[bs, :, rt_idx, 0] == 1)
        #         plt.plot(route_feat[bs, :, rt_idx, 1][ss_valid_mask].cpu(), route_feat[bs, :, rt_idx, 2][ss_valid_mask].cpu(), "ko")

        #     for ref_idx, ref_lines in enumerate(position[bs]):
        #         ref_valid_mask = valid_mask[bs, ref_idx]
        #         plt.plot(ref_lines[:, 0][ref_valid_mask].cpu(), ref_lines[:, 1][ref_valid_mask].cpu(), 'y-')
        #     plt.savefig(f"/home/jyyun/workshop/LatentDriver/vis/scene/{bs}_scene.png")
        #     plt.close()

        out, action_preds,rep_dist, latent_dist = self.forward(ss, reference_lines) #, position, vector, orientation, valid_mask, future_projection)

        trajectory, probability, prediction = (
            out["trajectory"][:B],
            out["probability"][:B],
            out["prediction"][:B],
        )
        ref_free_trajectory = out.get("ref_free_trajectory", None)

        # vehicle_obs = jnp.concatenate([xy, width, length, yaw, vel_xy, acc_xy, valid_mask], axis=-1)
        
        target = torch.cat((sdc_gt.unsqueeze(1), agent_gt), dim=1)
        target_heading = target[..., 4:5]
        targets_pos = torch.cat((target[..., :2], target_heading), dim=-1)
        valid_mask = target[..., -1]
        targets_vel = target[..., 5:7]

        target = torch.cat(
            [
                targets_pos[..., :2],
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )

        ego_reg_loss, ego_cls_loss = self.get_planning_loss(
            reference_lines, trajectory, probability.to(torch.float32), valid_mask[:, 0], target[:, 0], B
        )
        # ego_reg_loss, ego_cls_loss = ego_reg_loss.new_zeros(1), ego_reg_loss.new_zeros(1)
        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:B],
                target[:, 0, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask[:, 0]
            ).sum() / valid_mask[:, 0].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        idx = torch.arange(prediction.shape[1], device=prediction.device).unsqueeze(0).expand(B, -1)
        av_index = torch.where(ss['vehicle_segments'][:, :, 0] == 4)[1]
        idx = idx[idx != av_index.unsqueeze(1)].view(B, prediction.shape[1] - 1)
        not_av_idx = idx[:, :, None, None].expand(-1, -1, prediction.shape[2], prediction.shape[3])    
        prediction = prediction.gather(1, not_av_idx)
        
        prediction_loss = self.get_prediction_loss(
            prediction, valid_mask[:, 1:], target[:, 1:]
        )
        # prediction_loss = ego_reg_loss.new_zeros(1)

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + prediction_loss
            + ego_ref_free_reg_loss
        )
        logs = {}
        logs.update({
            'Lr/lr': self.optimizers().param_groups[0]['lr'],
            'loss/all': loss,
            'loss/ego_reg_loss': ego_reg_loss.item(),
            'loss/ego_cls_loss': ego_cls_loss.item(),
            'loss/prediction_loss': prediction_loss.item(),
            'loss/ego_ref_free_reg_loss': ego_ref_free_reg_loss.item(),
            })
        self.log_dict(logs,on_step=True)
        return loss

    def configure_optimizers(self):
        self.optim_conf.update(dict(lr = self.lr))
        optimizer = build_optimizer(self.optim_conf, self)
        all_steps = self.trainer.estimated_stepping_batches
        print(f'All step is {all_steps}')
        if self.sched_conf.type == 'CosineAnnealingLR':
            self.sched_conf.update(dict(T_max=all_steps))
        elif self.sched_conf.type == 'LinearLR':
            self.sched_conf.update(dict(total_iters=all_steps))
        elif self.sched_conf.type == 'OneCycleLR':
            self.sched_conf.update(dict(total_steps=all_steps))
        elif self.sched_conf.type == 'ConstantLR':
            self.sched_conf.update(dict(total_iters=all_steps))

        scheduler = build_scheduler(self.sched_conf,optimizer)
        scheduler = {
            'scheduler': scheduler, # The LR scheduler instance (required)
            'interval': 'step', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
        }
        return [optimizer], [scheduler]
