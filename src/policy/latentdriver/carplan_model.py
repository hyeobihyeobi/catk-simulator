import torch
import torch.nn as nn
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

from .layers.mlp_layer import MLPLayer

import torch.nn.functional as F

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
    #     self,
    #     ss, # bs , seq_len, state_attributes, state_dim
    #     position = None,# bs , seq_len, act_dim
    #     vector = None,
    #     orientation = None,
    #     valid_mask = None,
    #     future_projection = None,
    # ):
        self,
        ss,
        actions,
        reference_lines,
        timesteps,
        padding_mask,
    ):
        # reference_lines = {
        #     "position": position,
        #     "vector": vector,
        #     "orientation": orientation,
        #     "valid_mask": valid_mask,
        #     "future_projection": future_projection,
        # }
        batch_size, seq_length, state_elements, state_dims = ss.shape[0], ss.shape[1], ss.shape[2], ss.shape[3]
        flattened_states = ss.reshape(batch_size * seq_length, state_elements, state_dims)
        bert_embeddings = self._encode_with_bert(flattened_states)
        bert_embeddings = bert_embeddings.reshape(batch_size, seq_length, -1, self.bert.hidden_size)

        actions_layers = []
        cur_latent_token = bert_embeddings[:,:,0:1,:]
        B,T,_,_ = bert_embeddings.shape
        # query_pe = self.query_pe(None).repeat(B*T,1,1).permute(1,0,2)
        # query_content = torch.zeros_like(query_pe)
        # action_dis = None
        # fut_latent_dis = None
        latent_dist = None
        rep_dist = None

        current_bert_embeddings = bert_embeddings[:, 1, 1:]

        input_batch_type = flattened_states[..., 0]

        car_mask = (input_batch_type == 2).unsqueeze(-1)
        road_graph_mask = (input_batch_type == 3).unsqueeze(-1)
        route_mask = (input_batch_type == 1).unsqueeze(-1)
        sdc_mask = (input_batch_type == 4).unsqueeze(-1)
        # current_bert_padding_mask = (~torch.logical_or(torch.logical_or(torch.logical_or(route_mask, car_mask), road_graph_mask), sdc_mask)).squeeze(-1).reshape(batch_size, T, -1)[:, 1]
        current_bert_padding_mask = (~torch.logical_or(torch.logical_or(route_mask, car_mask), road_graph_mask)).squeeze(-1).reshape(batch_size, T, -1)[:, 1]

        agent_embeddings_for_prediction = current_bert_embeddings[:, 20:148] + bert_embeddings[:, 1, 0:1]
        loc = self.loc_predictor(agent_embeddings_for_prediction).view(B, 128, 80, 2)
        yaw = self.yaw_predictor(agent_embeddings_for_prediction).view(B, 128, 80, 2)
        vel = self.vel_predictor(agent_embeddings_for_prediction).view(B, 128, 80, 2)
        prediction = torch.cat([loc, yaw, vel], dim=-1)

        ref_line_available = reference_lines["position"].shape[1] > 0
        R, M = reference_lines["position"].shape[1], 12

        if ref_line_available:
            trajectory, probability, pred_scenario_type, q, tgt_route, gates, load, gates_dict, dist_predictions, scores_list = self.planning_decoder(
                reference_lines, {"enc_emb": current_bert_embeddings, "enc_key_padding_mask": current_bert_padding_mask, "enc_pos": None, "x_scene_encoder": None}
            )
        else:
            trajectory, probability, pred_scenario_type, q, tgt_route, gates, load, gates_dict,scores_list = None, None, None, None, None, None, None, None, None

        # if self.ref_free_traj:
        ref_free_traj = self.ref_free_decoder(bert_embeddings[:, 1, 0]).reshape(
            batch_size, 80, 4
        )

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
            "load": load,
            "scores_list": scores_list,
            "ref_free_trajectory": ref_free_traj,
        }

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

#         return out, best_trajectory,latent_dist, rep_dist
        return out, output_ref_free_trajectory,latent_dist, rep_dist

    def get_predictions(
        self, states, actions, reference_lines, timesteps, num_envs=1, **kwargs
    ):
        states_elements = states.shape[2]

        # max_length is the context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1)) #(4, 2)

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            states_elements,
                            states.shape[-1],
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32) #(B, 2, N, 7) #과거 정보가 없으면 0으로 처리
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        # import pdb; pdb.set_trace()
        for bs in range(num_envs):
            for ref_idx, ref_lines in enumerate(reference_lines["position"][bs]):
                ref_valid_mask = reference_lines["valid_mask"][bs, ref_idx]
                ref_lines_valid = ref_lines[ref_valid_mask]
                diffs = ref_lines_valid[1:] - ref_lines_valid[:-1]           # (N-1, 2)
                dists = torch.norm(diffs, dim=1)             # 각 구간 거리
                total_length = dists.sum()
                if total_length > 120:
                    reference_lines["valid_mask"][bs, ref_idx] = False

        _, action, _, _ = self.forward(
            states,
            actions,
            reference_lines,
            timesteps,
            padding_mask=padding_mask,
        )
        B, T = states.shape[0], states.shape[1]
        # if self.function in ['enc-LWM-MPP','enc-MPP']:
        #     _, _, _, action_ = unpack_action(action[-1].reshape(B*T,-1,7), B, T)
        #     action = action_.reshape(B, T, -1)
        return action

    def get_planning_loss(self, future_projection, valid_mask, trajectory, probability, target_valid_mask, target, bs):
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        num_valid_points = target_valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~(valid_mask[:bs].any(-1))  # (bs, R)
        unvalid_batch_mask = r_padding_mask.all(-1)
        future_projection = future_projection[:bs][
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
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def training_step(self, batch, batch_idx):

        (ss, position, vector, orientation, valid_mask, future_projection, target, target_vel, target_valid_mask, is_sdc) = batch
        B, T, _, _ = ss.shape

        for bs in range(B):
            for ref_idx, ref_lines in enumerate(position[bs]):
                ref_valid_mask = valid_mask[bs, ref_idx]
                ref_lines_valid = ref_lines[ref_valid_mask]
                diffs = ref_lines_valid[1:] - ref_lines_valid[:-1]           # (N-1, 2)
                dists = torch.norm(diffs, dim=1)             # 각 구간 거리
                total_length = dists.sum()
                if total_length > 120:
                    valid_mask[bs, ref_idx] = False

        import matplotlib.pyplot as plt
        route_feat = ss[:, :, :20]
        agent_feat = ss[:, :, 20:148]
        road_feat = ss[:, :, 148:]

        agent_mask = (ss[:, 1, 20:148, 0] == 2)

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

        out, action_preds,rep_dist, latent_dist = self.forward(ss, position, vector, orientation, valid_mask, future_projection)

        trajectory, probability, prediction = (
            out["trajectory"][:B],
            out["probability"][:B],
            out["prediction"][:B],
        )
        ref_free_trajectory = out.get("ref_free_trajectory", None)

        targets_pos = target
        # target_valid_mask = target_valid_mask
        targets_vel = target_vel

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
            future_projection, valid_mask, trajectory, probability.to(torch.float32), target_valid_mask[is_sdc], target[is_sdc], B
        )
        # ego_reg_loss, ego_cls_loss = ego_reg_loss.new_zeros(1), ego_reg_loss.new_zeros(1)
        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:B],
                target[is_sdc][:, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * target_valid_mask[is_sdc]
            ).sum() / target_valid_mask[is_sdc].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        prediction_loss = self.get_prediction_loss(
            prediction[agent_mask], target_valid_mask[agent_mask], target[agent_mask]
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
