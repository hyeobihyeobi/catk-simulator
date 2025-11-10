import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Callable, Tuple
from omegaconf import OmegaConf

from waymax import datatypes
from waymax.agents import catk_sim_agent
from waymax.datatypes.constant import TIME_INTERVAL

from waymax.agents.catk.smart.model.smart import SMART  # kept for reference
from waymax.agents.catk.smart.modules.smart_decoder import SMARTDecoder
from waymax.agents.catk.smart.tokens.token_processor import TokenProcessor


def _to_torch(x) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, 'dtype') and 'jax' in type(x).__module__:
        x = jax.device_get(x)
    x = jnp.array(x)
    t = torch.from_numpy(x)
    return t


def _object_type_to_catk_types(obj_types_np: jnp.ndarray) -> torch.Tensor:
    # Waymax ObjectTypeIds: UNSET=0, VEHICLE=1, PEDESTRIAN=2, CYCLIST=3
    m = jnp.zeros_like(obj_types_np)
#     m[obj_types_np == 1] = 0  # veh
#     m[obj_types_np == 2] = 1  # ped
#     m[obj_types_np == 3] = 2  # cyc

    m = jnp.where(obj_types_np == 1, 0, m)
    m = jnp.where(obj_types_np == 2, 1, m)
    m = jnp.where(obj_types_np == 3, 2, m)

    m = m.astype(jnp.int64)
    return torch.from_numpy(jax.device_get(m).copy())
#     return torch.from_numpy(m.astype(jnp.int64))


class CATK_Simulator(catk_sim_agent.CATK_SimAgentActor):
    """CATK Simulator class."""

    def __init__(
        self,
        is_controlled_func: Optional[
            Callable[[datatypes.SimulatorState], jax.Array]
        ] = None,
        invalidate_on_end: bool = False,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__(is_controlled_func=is_controlled_func)
        self.invalidate_on_end = invalidate_on_end

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint config and weights
        if ckpt_path is None:
            root_path = os.environ.get('ROOT_PATH')
            if not root_path:
                raise ValueError(
                    'CATK_Simulator requires ROOT_PATH environment variable when ckpt_path is not provided'
                )
            ckpt_path = os.path.join(
                root_path, 'checkpoints', 'catk_trained_parameter', 'pre_bc_E31.ckpt'
            )
        ckpt = torch.load(ckpt_path, map_location='cpu')
        mc = ckpt['hyper_parameters']['model_config']
        self.model_config = mc

        # Build token processor and encoder
        self.token_processor = TokenProcessor(**mc['token_processor']).to(self.device)
        self.encoder = SMARTDecoder(
            n_token_agent=self.token_processor.n_token_agent,
            **mc['decoder'],
        ).to(self.device)
        self.encoder.eval()
        self.token_processor.eval()

        # Load weights: strip leading 'encoder.'
        enc_sd = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('encoder.'):
                enc_sd[k[len('encoder.'):]] = v
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        # Silence warnings in runtime; we only need decoder weights

        # Sampling scheme
        self.sampling_scheme = mc['validation_rollout_sampling']

    def update_trajectory(
        self, state: datatypes.SimulatorState
    ) -> datatypes.TrajectoryUpdate:
        """Returns a trajectory update of shape (..., num_objects, 1)."""
        next_traj = self._get_next_trajectory_by_projection(
            state,
        )
        return datatypes.TrajectoryUpdate(
            x=next_traj.x,
            y=next_traj.y,
            yaw=next_traj.yaw,
            vel_x=next_traj.vel_x,
            vel_y=next_traj.vel_y,
            valid=next_traj.valid,
        )

    def _build_map_tokens(self, state: datatypes.SimulatorState) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        rg = state.roadgraph_points

        def _reshape(arr, trailing_dims: int) -> np.ndarray:
            arr_np = np.array(jax.device_get(arr))
            leading = arr_np.shape[:-trailing_dims]
            if len(leading) == 0:
                return arr_np.reshape((1,) + arr_np.shape[-trailing_dims:])
            return arr_np.reshape((-1,) + arr_np.shape[-trailing_dims:])

        pos_xy = _reshape(rg.xy, 2)
        dir_xy = _reshape(rg.dir_xy, 2)
        types = _reshape(rg.types, 1)
        valid = _reshape(rg.valid, 1)

        n_graphs = pos_xy.shape[0]
        traj_pos_list: list[torch.Tensor] = []
        traj_theta_list: list[torch.Tensor] = []
        pt_type_list: list[torch.Tensor] = []
        batch_list: list[torch.Tensor] = []

        for graph_idx in range(n_graphs):
            mask = valid[graph_idx].astype(bool)
            pos = pos_xy[graph_idx][mask]
            direction = dir_xy[graph_idx][mask]
            type_ids = types[graph_idx][mask]

            if pos.shape[0] == 0:
                traj_pos = torch.zeros((1, 3, 2), dtype=torch.float32)
                traj_theta = torch.zeros((1,), dtype=torch.float32)
                pt_type = torch.zeros((1,), dtype=torch.long)
            else:
                dir_norm = np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6
                dir_u = direction / dir_norm
                d = 1.0
                p0 = pos - d * dir_u
                p1 = pos
                p2 = pos + d * dir_u
                traj_pos_np = np.stack([p0, p1, p2], axis=1).astype(np.float32)
                traj_theta_np = np.arctan2(dir_u[:, 1], dir_u[:, 0]).astype(np.float32)

                traj_pos = torch.from_numpy(traj_pos_np.copy())
                traj_theta = torch.from_numpy(traj_theta_np.copy())
                pt_type = torch.from_numpy(type_ids.astype(np.int64).copy())

            traj_pos_list.append(traj_pos)
            traj_theta_list.append(traj_theta)
            pt_type_list.append(pt_type)
            batch_list.append(torch.full((traj_pos.shape[0],), graph_idx, dtype=torch.long))

        traj_pos = torch.cat(traj_pos_list, dim=0)
        traj_theta = torch.cat(traj_theta_list, dim=0)
        pt_type = torch.cat(pt_type_list, dim=0)
        batch = torch.cat(batch_list, dim=0)
        return traj_pos, traj_theta, pt_type, batch

    def _state_to_heterodata(self, state: datatypes.SimulatorState) -> HeteroData:
        # Build agent tensors from trajectories (host arrays)
        log_traj = state.log_trajectory
        sim_traj = state.sim_trajectory
        meta = state.object_metadata

        def _reshape_traj(arr) -> np.ndarray:
            arr_np = np.array(jax.device_get(arr))
            leading = arr_np.shape[:-2]
            if len(leading) == 0:
                arr_np = arr_np.reshape((1,) + arr_np.shape[-2:])
            else:
                arr_np = arr_np.reshape((-1,) + arr_np.shape[-2:])
            return arr_np

        def _reshape_meta(arr) -> np.ndarray:
            arr_np = np.array(jax.device_get(arr))
            leading = arr_np.shape[:-1]
            if len(leading) == 0:
                arr_np = arr_np.reshape((1, arr_np.shape[-1]))
            else:
                arr_np = arr_np.reshape((-1, arr_np.shape[-1]))
            return arr_np

        log_x = _reshape_traj(log_traj.x)
        log_y = _reshape_traj(log_traj.y)
        log_yaw = _reshape_traj(log_traj.yaw)
        log_vel_x = _reshape_traj(log_traj.vel_x)
        log_vel_y = _reshape_traj(log_traj.vel_y)
        log_valid = _reshape_traj(log_traj.valid)

        sim_x = _reshape_traj(sim_traj.x)
        sim_y = _reshape_traj(sim_traj.y)
        sim_yaw = _reshape_traj(sim_traj.yaw)
        sim_vel_x = _reshape_traj(sim_traj.vel_x)
        sim_vel_y = _reshape_traj(sim_traj.vel_y)
        sim_valid = _reshape_traj(sim_traj.valid)

        n_graphs, n_agent_per_graph, n_step = log_x.shape
        n_agent_total = n_graphs * n_agent_per_graph

        timesteps = np.array(jax.device_get(state.timestep))
        if timesteps.shape == ():
            timesteps = timesteps.reshape(1)
        else:
            timesteps = timesteps.reshape(-1)
        if timesteps.size != n_graphs:
            timesteps = np.broadcast_to(timesteps.reshape(-1), (n_graphs,))
        timesteps = np.clip(timesteps.astype(np.int32), 0, n_step - 1)

        def _merge_history(log_arr: np.ndarray, sim_arr: np.ndarray) -> np.ndarray:
            merged = log_arr.copy()
            for graph_idx, t in enumerate(timesteps):
                merged[graph_idx, :, : t + 1] = sim_arr[graph_idx, :, : t + 1]
            return merged

        merged_x = _merge_history(log_x, sim_x)
        merged_y = _merge_history(log_y, sim_y)
        merged_yaw = _merge_history(log_yaw, sim_yaw)
        merged_vel_x = _merge_history(log_vel_x, sim_vel_x)
        merged_vel_y = _merge_history(log_vel_y, sim_vel_y)
        merged_valid = _merge_history(log_valid, sim_valid)

        hist_shift = getattr(self.token_processor, "shift", 5)
        start_indices = np.clip(timesteps - hist_shift, 0, n_step - 1)

        def _time_shift(arr: np.ndarray) -> np.ndarray:
            shifted = np.empty_like(arr)
            for graph_idx, start in enumerate(start_indices):
                if start <= 0:
                    shifted[graph_idx] = arr[graph_idx]
                else:
                    remaining = arr.shape[-1] - start
                    shifted[graph_idx, :, :remaining] = arr[graph_idx, :, start:]
                    shifted[graph_idx, :, remaining:] = arr[graph_idx, :, -1:,]
            return shifted

        merged_x = _time_shift(merged_x)
        merged_y = _time_shift(merged_y)
        merged_yaw = _time_shift(merged_yaw)
        merged_vel_x = _time_shift(merged_vel_x)
        merged_vel_y = _time_shift(merged_vel_y)
        merged_valid = _time_shift(merged_valid)

        if not hasattr(self, "_log_shape_debug"):
            print("[CATK] log.x shape:", log_x.shape)
            self._log_shape_debug = True

        # XY only for tokenizer inputs
        pos_xy = np.stack([merged_x, merged_y], axis=-1).astype(np.float32)
        vel_xy = np.stack([merged_vel_x, merged_vel_y], axis=-1).astype(np.float32)
        heading = merged_yaw.astype(np.float32)
        valid_mask = merged_valid.astype(np.bool_)

        # Metadata
        obj_types = _reshape_meta(meta.object_types)
        is_sdc = _reshape_meta(meta.is_sdc).astype(bool)
        if hasattr(meta, 'objects_of_interest'):
            ooi = _reshape_meta(meta.objects_of_interest).astype(bool)
        else:
            ooi = np.zeros_like(is_sdc, dtype=bool)
        ids = _reshape_meta(meta.ids)

        agent_type = _object_type_to_catk_types(jnp.array(obj_types)).reshape(-1)
        agent_id = torch.from_numpy(ids.astype(np.int64).reshape(-1))
        role = torch.stack([
            torch.from_numpy(is_sdc.astype(np.bool_).reshape(-1)),
            torch.from_numpy(ooi.astype(np.bool_).reshape(-1)),
            torch.from_numpy((~is_sdc & ~ooi).astype(np.bool_).reshape(-1)),
        ], dim=-1)

        # Use current step geometry from current_sim_trajectory (already dynamically sliced in simulator)
        cur = state.current_sim_trajectory
        cur_width_np = np.array(jax.device_get(cur.width))
        if not hasattr(self, "_cur_shape_debug"):
            print("[CATK] cur.width shape:", cur_width_np.shape)
            self._cur_shape_debug = True
        cur_width_np = cur_width_np.reshape(n_graphs, n_agent_per_graph, -1).astype(np.float32)[..., 0]
        cur_length_np = np.array(jax.device_get(cur.length)).reshape(n_graphs, n_agent_per_graph, -1).astype(np.float32)[..., 0]
        cur_height_np = np.array(jax.device_get(cur.height)).reshape(n_graphs, n_agent_per_graph, -1).astype(np.float32)[..., 0]
        cur_width = torch.from_numpy(cur_width_np.reshape(-1))
        cur_length = torch.from_numpy(cur_length_np.reshape(-1))
        cur_height = torch.from_numpy(cur_height_np.reshape(-1))
        shape = torch.stack([
            cur_width,
            cur_length,
            cur_height,
        ], dim=-1)

        # Build HeteroData
        data = HeteroData()
        data['agent'].num_nodes = int(n_agent_total)
        data['agent']['valid_mask'] = torch.from_numpy(
            valid_mask.reshape(n_graphs, n_agent_per_graph, n_step).reshape(n_agent_total, n_step)
        )
        data['agent']['role'] = role                                          # [n_agent, 3]
        data['agent']['id'] = agent_id                                        # [n_agent]
        data['agent']['type'] = agent_type                                    # [n_agent]
        data['agent']['position'] = torch.from_numpy(
            pos_xy.reshape(n_graphs, n_agent_per_graph, n_step, 2).reshape(n_agent_total, n_step, 2)
        )
        data['agent']['heading'] = torch.from_numpy(
            heading.reshape(n_graphs, n_agent_per_graph, n_step).reshape(n_agent_total, n_step)
        )
        data['agent']['velocity'] = torch.from_numpy(
            vel_xy.reshape(n_graphs, n_agent_per_graph, n_step, 2).reshape(n_agent_total, n_step, 2)
        )
        data['agent']['shape'] = shape                                        # [n_agent, 3]
        data['agent']['batch'] = torch.arange(n_graphs, dtype=torch.long).repeat_interleave(n_agent_per_graph)

        # Map tokens (already robust to empty)
        traj_pos, traj_theta, pt_type, batch = self._build_map_tokens(state)
        data['map_save'].num_nodes = int(traj_pos.shape[0])
        data['map_save']['traj_pos'] = traj_pos                               # [n_pl, 3, 2]
        data['map_save']['traj_theta'] = traj_theta                           # [n_pl]
        data['pt_token'].num_nodes = int(traj_pos.shape[0])
        data['pt_token']['type'] = pt_type
        data['pt_token']['pl_type'] = torch.zeros_like(pt_type)
        data['pt_token']['light_type'] = torch.zeros_like(pt_type)
        data['pt_token']['batch'] = batch

        data.num_graphs = n_graphs
        return data, timesteps

    @staticmethod
    def _shift_agent_sequences(tokenized_agent: Dict[str, torch.Tensor], shift_counts: torch.Tensor) -> None:
        """Aligns agent token sequences to the latest timestep."""

        def _shift_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim < 2:
                return tensor
            n_agent, t_steps = tensor.shape[:2]
            device = tensor.device
            shifts = shift_counts.to(device)
            tensor = tensor.clone()
            for agent_idx in range(n_agent):
                shift = int(shifts[agent_idx].item())
                if shift <= 0:
                    continue
                shift = min(shift, t_steps - 1)
                body = tensor[agent_idx, shift:]
                pad = tensor[agent_idx, -1:].expand(shift, *tensor.shape[2:])
                tensor[agent_idx] = torch.cat([body, pad], dim=0)
            return tensor

        seq_keys = [
            "gt_pos",
            "gt_heading",
            "gt_idx",
            "sampled_idx",
            "sampled_pos",
            "sampled_heading",
            "gt_pos_raw",
            "gt_head_raw",
            "gt_valid_raw",
            "valid_mask",
        ]
        for key in seq_keys:
            tensor = tokenized_agent.get(key)
            if isinstance(tensor, torch.Tensor):
                tokenized_agent[key] = _shift_tensor(tensor)

    def _get_next_trajectory_by_projection(
        self,
        state: datatypes.SimulatorState,
    ) -> datatypes.Trajectory:
        """Run CAT-K to get the next trajectory for all agents."""
        if self.invalidate_on_end and bool(jax.device_get(state.is_done())):
            # Return invalid update
            cur = state.current_sim_trajectory
            invalid = jnp.zeros_like(cur.valid, dtype=jnp.bool_)
            return datatypes.Trajectory(
                x=cur.x, y=cur.y, z=cur.z, yaw=cur.yaw,
                vel_x=cur.vel_x, vel_y=cur.vel_y, valid=invalid,
                timestamp_micros=cur.timestamp_micros,
                length=cur.length, width=cur.width, height=cur.height,
            )

        data, timesteps = self._state_to_heterodata(state)
        tokenized_map, tokenized_agent = self.token_processor(data)

        hist_shift = self.token_processor.shift
        step_current_10hz = self.encoder.agent_encoder.num_historical_steps - 1
        step_current_2hz = max(step_current_10hz // hist_shift, 1)
        timesteps_2hz = np.floor_divide(timesteps, hist_shift)
        max_shift = tokenized_agent["gt_pos"].shape[1] - 1
        n_agent_per_graph = tokenized_agent["gt_pos"].shape[0] // data.num_graphs
        shift_graph = np.clip(timesteps_2hz - step_current_2hz, 0, max_shift)
        shift_counts = torch.from_numpy(shift_graph.astype(np.int64)).repeat_interleave(n_agent_per_graph)
        self._shift_agent_sequences(tokenized_agent, shift_counts)

        sim_xy = np.array(jax.device_get(state.sim_trajectory.xy))
        sim_yaw = np.array(jax.device_get(state.sim_trajectory.yaw))
        sim_valid = np.array(jax.device_get(state.sim_trajectory.valid))
        sim_xy = sim_xy.reshape(data.num_graphs, n_agent_per_graph, -1, 2)
        sim_yaw = sim_yaw.reshape(data.num_graphs, n_agent_per_graph, -1)
        sim_valid = sim_valid.reshape(data.num_graphs, n_agent_per_graph, -1)

        hist_pos = np.zeros((tokenized_agent["gt_pos"].shape[0], step_current_2hz, 2), dtype=np.float32)
        hist_head = np.zeros((tokenized_agent["gt_heading"].shape[0], step_current_2hz), dtype=np.float32)
        hist_valid = np.zeros((tokenized_agent["valid_mask"].shape[0], step_current_2hz), dtype=bool)
        for graph_idx in range(data.num_graphs):
            base = graph_idx * n_agent_per_graph
            for agent_idx in range(n_agent_per_graph):
                seq_idx = base + agent_idx
                for hist_idx in range(step_current_2hz):
                    target = timesteps[graph_idx] - (step_current_2hz - hist_idx) * hist_shift + hist_shift
                    target = int(np.clip(target, 0, sim_xy.shape[2] - 1))
                    hist_pos[seq_idx, hist_idx] = sim_xy[graph_idx, agent_idx, target]
                    hist_head[seq_idx, hist_idx] = sim_yaw[graph_idx, agent_idx, target]
                    hist_valid[seq_idx, hist_idx] = sim_valid[graph_idx, agent_idx, target]

        tokenized_agent["gt_pos"][:, :step_current_2hz] = torch.from_numpy(hist_pos)
        tokenized_agent["gt_heading"][:, :step_current_2hz] = torch.from_numpy(hist_head)
        tokenized_agent["valid_mask"][:, :step_current_2hz] = torch.from_numpy(hist_valid)
        # Move to device
        tokenized_map = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokenized_map.items()
        }
        tokenized_agent = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokenized_agent.items()
        }

        with torch.no_grad():
            pred = self.encoder.inference(tokenized_map, tokenized_agent, self.sampling_scheme)
        pred_xy_10hz = pred.get('pred_traj_10hz')  # [n_agent, 80, 2]
        pred_head_10hz = pred.get('pred_head_10hz')  # [n_agent, 80]
        n_graphs = tokenized_agent["num_graphs"]
        n_agent_per_graph = tokenized_agent["type"].shape[0] // n_graphs

        # Current step xy for vel
        cur = state.current_sim_trajectory
        cur_x = jax.device_get(cur.x)[0, :, :, 0]
        cur_y = jax.device_get(cur.y)[0, :, :, 0]

        # Next step
        if pred_xy_10hz is None:
            # Fallback: keep current
            next_x = torch.from_numpy(cur_x.astype(jnp.float32))
            next_y = torch.from_numpy(cur_y.astype(jnp.float32))
            next_head = torch.from_numpy(
                jax.device_get(cur.yaw)[0, :, :, 0].astype(jnp.float32)
            )
        else:
            next_x = pred_xy_10hz[:, 0, 0].cpu().reshape(n_graphs, n_agent_per_graph)
            next_y = pred_xy_10hz[:, 0, 1].cpu().reshape(n_graphs, n_agent_per_graph)
            if pred_head_10hz is not None:
                next_head = pred_head_10hz[:, 0].cpu().reshape(
                    n_graphs, n_agent_per_graph
                )
            else:
                # Approximate from displacement if no head
                dx = next_x - torch.from_numpy(cur_x)
                dy = next_y - torch.from_numpy(cur_y)
                next_head = torch.atan2(dy, dx)

        # Velocity
        vx = (next_x - torch.from_numpy(cur_x)) / TIME_INTERVAL
        vy = (next_y - torch.from_numpy(cur_y)) / TIME_INTERVAL

        if not hasattr(self, "_traj_shape_debug"):
            print("[CATK] cur.x shape:", jax.device_get(cur.x).shape)
            self._traj_shape_debug = True
        base_shape = jax.device_get(cur.x).shape
        new_shape = base_shape
        x = jnp.asarray(next_x.numpy(), dtype=jnp.float32).reshape(new_shape)
        y = jnp.asarray(next_y.numpy(), dtype=jnp.float32).reshape(new_shape)
        yaw = jnp.asarray(next_head.numpy(), dtype=jnp.float32).reshape(new_shape)
        vx = jnp.asarray(vx.numpy(), dtype=jnp.float32).reshape(new_shape)
        vy = jnp.asarray(vy.numpy(), dtype=jnp.float32).reshape(new_shape)
        valid_mask = ~jax.device_get(state.object_metadata.is_sdc)
        valid = jnp.asarray(valid_mask, dtype=bool).reshape(new_shape)

        cur = state.current_sim_trajectory
        return datatypes.Trajectory(
            x=x, y=y, z=cur.z, yaw=yaw,
            vel_x=vx, vel_y=vy, valid=valid,
            timestamp_micros=cur.timestamp_micros,
            length=cur.length, width=cur.width, height=cur.height,
        )
