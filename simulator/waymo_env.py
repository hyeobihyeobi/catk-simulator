
import numpy as np
import os
import gymnasium as gym

import jax
import jax.numpy as jnp
from waymax import dynamics
from simulator.waymo_base import WaymoBaseEnv
from simulator.metric import Metric
from simulator.utils import combin_traj,build_discretizer,get_cache_polylines_baseline
from simulator.observation import (
    get_obs_from_routeandmap_saved_jit,
    get_obs_from_routeandmap_saved_pmap,
    preprocess_data_dist_jnp,
)
from simulator.observation import get_obs_from_routeandmap_saved

#JY
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import LineString, Point
import warnings

import torch
import numpy

def to_tensor(data):
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, numpy.ndarray):
        if data.dtype == numpy.float64:
            return torch.from_numpy(data).float()
        else:
            return torch.from_numpy(data)
    elif isinstance(data, numpy.number):
        return torch.tensor(data).float()
    elif isinstance(data, list):
        return data
    elif isinstance(data, int):
        return torch.tensor(data)
    elif isinstance(data, tuple):
        return to_tensor(data[0])
    elif isinstance(data, str):
        return data
    else:
        print(type(data), data)
        raise NotImplementedError

def resample_and_pad_zeros(path_xy: np.ndarray, target_n: int = 120, ds: float = 1.0):
    """
    path_xy : (N,2)  원본 경로
    target_n: 출력 길이 (예: 120)
    ds      : 샘플 간격(m)
    return  : (resampled_path, valid_mask)
              resampled_path: (target_n,2) — 1m 간격 보간 + 0패딩
              valid_mask: (target_n,) — 유효한 보간 구간(True)/패딩(False)
    """
    P = np.asarray(path_xy, float)
    if len(P) < 2:
        out = np.zeros((target_n, 2))
        mask = np.zeros(target_n, dtype=bool)
        return out, mask

    # 누적 거리
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    L = s[-1]

    # 목표 샘플 위치 (0, 1, 2, ..., target_n-1) * ds
    q = ds * np.arange(target_n)
    valid_mask = q <= L

    out = np.zeros((target_n, 3), dtype=float)
    
    if out[valid_mask].shape[0] < 2:
        out = np.zeros((target_n, 2))
        mask = np.zeros(target_n, dtype=bool)
        return out, mask
    
    if np.any(valid_mask):
        out[valid_mask, 0] = np.interp(q[valid_mask], s, P[:, 0])
        out[valid_mask, 1] = np.interp(q[valid_mask], s, P[:, 1])
        
        xy = out[valid_mask, :2]
        diffs = np.diff(xy, axis=0)
        yaw = np.arctan2(diffs[:,1], diffs[:,0])
        yaw = np.concatenate([yaw, yaw[-1:]])
        # try:
        out[valid_mask, 2] = yaw
        # except:
        #     print("hrer")

    return out, valid_mask

def shortest_path_unordered(valid_points, av_xy, k_init=6, k_max=64):
    """
    valid_points: (N,2) float, 순서 보장 없음. 마지막 점이 goal.
    av_xy       : (2,)   float, AV 현재 위치 (world 좌표)
    k_init      : 초기 k-NN 이웃 수
    k_max       : 실패 시 k를 두 배씩 늘려 이 값까지 재시도
    return      : path_xy (M,2), path_idx (M,)
    """
    P = np.asarray(valid_points, dtype=float)
    N = len(P)
    if N < 2:
        return P.copy(), np.arange(N)

    # 시작/목표 인덱스
    start = 0 #int(np.argmin(np.einsum('ij,j->i',(P - av_xy), (P - av_xy).T.diagonal()*0 + 1)))  # 빠른 L2 argmin
    goal  = N - 1

    # k를 점차 늘리며 연결 시도
    k = min(k_init, max(1, N-1))
    while True:
        # k-NN 그래프 구성 (자기자신 제외)
        tree = cKDTree(P)
        k_eff = min(k+1, N)  # 자기자신 포함해서 k+1
        dists, nbrs = tree.query(P, k=k_eff)
        rows = np.repeat(np.arange(N), k_eff-1)
        cols = nbrs[:, 1:].ravel()
        w    = dists[:, 1:].ravel()

        # 희소 인접행렬(무방향 대칭)
        A = coo_matrix((w, (rows, cols)), shape=(N, N))
        A = A.minimum(A.T)

        # Dijkstra 최단경로
        dist, pred = dijkstra(A, indices=start, return_predecessors=True)
        if np.isfinite(dist[goal]):
            # 경로 복원
            path_idx = []
            u = goal
            while u != -9999 and u != start:
                path_idx.append(u)
                u = pred[u]
            path_idx.append(start)
            path_idx = path_idx[::-1]
            return P[path_idx], np.array(path_idx, dtype=int)

        # 실패하면 k 증가 후 재시도
        if k >= k_max:
            # import matplotlib.pyplot as plt
            # plt.plot(valid_points[:, 0], valid_points[:, 1], 'r-')
            # plt.xlim(100, 300)
            # plt.ylim(600, 800)
            # plt.savefig("/home/jyyun/workshop/LatentDriver/vis/error.png")
            # plt.close()
            return valid_points, None
            # raise RuntimeError("경로를 찾지 못했습니다. 포인트 분포가 끊겨 있거나 반경/이웃 수가 부족합니다.")
        k = min(k*2, k_max)

def get_agents_gt(state):
    for de in range(state.shape[0]):
        target_list = []
        target_vel_list = []
        target_valid_list = []
        target_is_sdc_list = []
        for bs in range(state.shape[1]):
            agents_future_xy = state.log_trajectory.xy[de, bs, :, 10:]
            agents_future_vel_xy = state.log_trajectory.vel_xy[de, bs, :, 10:]
            agents_future_yaw = state.log_trajectory.yaw[de, bs, :, 10:]
            agents_future_valid = state.log_trajectory.valid[de, bs, :, 10:]
            
            is_sdc_idx = state.object_metadata.is_sdc[de, bs]
            # ego_future_valid = state.log_trajectory.valid[de, bs, :, 11:][is_sdc_idx]
            # ego_future = ego_future[is_sdc_idx][ego_future_valid]
            av_curr_xy = state.current_log_trajectory.xy[de, bs][is_sdc_idx].squeeze(0)
            av_curr_yaw = state.current_log_trajectory.yaw[de, bs][is_sdc_idx].squeeze(0)
            
            # agents_curr_xy = state.current_log_trajectory.xy[de, bs]
            # agents_curr_yaw = state.current_log_trajectory.yaw[de, bs]
            
            rotate_mat = np.array(
                [
                    [np.cos(av_curr_yaw), -np.sin(av_curr_yaw)],
                    [np.sin(av_curr_yaw), np.cos(av_curr_yaw)],
                ],
                dtype=np.float64,
            ).squeeze(-1)
            
            agents_future_xy = np.matmul(agents_future_xy - av_curr_xy, rotate_mat)
            agents_future_vel_xy = np.matmul(agents_future_vel_xy, rotate_mat)
            agents_future_yaw = agents_future_yaw - av_curr_yaw
            
            ##AV-Centric
            agents_future_xy = agents_future_xy[:, 1:] - agents_future_xy[:, 0:1]
            agents_future_yaw = agents_future_yaw[:, 1:] - agents_future_yaw[:, 0:1]
            target = np.concatenate([agents_future_xy, agents_future_yaw[..., None]], -1)
            target[~agents_future_valid[:, 1:]] = 0
            
            target_list.append(target)
            target_valid_list.append(agents_future_valid[:, 1:])
            target_is_sdc_list.append(is_sdc_idx)
            target_vel_list.append(agents_future_vel_xy[:, 1:])
    
    return {
            "target": target_list,
            "target_vel": target_vel_list,
            "valid_mask": target_valid_list,
            "is_sdc": target_is_sdc_list
            }

def get_reference_line(state):
    d_position_list = []
    d_vector_list = []
    d_orientation = []
    d_valid_mask = []
    d_future_projection = []
    
    # import pdb; pdb.set_trace()
    for de in range(state.shape[0]):
        b_position_list = []
        b_vector_list = []
        b_orientation = []
        b_valid_mask = []
        b_future_projection = []
        for bs in range(state.shape[1]):
            reference_line_list = []
            reference_line_valid_mask_list = []
            on_route = np.array(state.sdc_paths.on_route)[de, bs].squeeze(1)
            on_route_valid = np.array(state.sdc_paths.valid)[de, bs][on_route]
            
            # import matplotlib.pyplot as plt
            for i in range(on_route_valid.shape[0]):
                
                if i >= 1:
                    break
                # num_steps = np.sum(np.array(state.sdc_paths.valid)[i, :])
                # assert np.all(np.array(state.sdc_paths.valid)[i, :num_steps])

                points = np.array(state.sdc_paths.xy[de, bs])[on_route][i][on_route_valid[i]]
                # points = scenario.sdc_paths.xy[i, :num_steps]
                av_xy = np.array(state.current_log_trajectory.xy)[de, bs][np.array(state.object_metadata.is_sdc)[de, bs]].squeeze(1)
                distances = np.sqrt(np.sum((points - av_xy)**2, axis=1))
                start_idx = np.argmin(distances) 
                if distances[start_idx] > 1:
                    continue
                start_idx = np.argmin(distances) 
                path_xy, path_idx = shortest_path_unordered(points[start_idx:], av_xy)
                path_xy, mask = resample_and_pad_zeros(path_xy, target_n=120)
                if path_xy[mask].shape[0] < 2:
                    continue
                reference_line_list.append(path_xy)
                reference_line_valid_mask_list.append(mask)
                
                # plt.plot(points[:, 0], points[:, 1], 'r-')
                # plt.plot(points[start_idx:][0, 0], points[start_idx:][0, 1], 'bo')
                # plt.savefig(f"/home/jyyun/workshop/LatentDriver/vis/origin_{i}_ref.png")
                # plt.close()
            
            remove_index = set()
            for i in range(len(reference_line_list)):
                for j in range(i + 1, len(reference_line_list)):
                    if j in remove_index:
                        continue
                    min_len = min(len(reference_line_list[i][reference_line_valid_mask_list[i]]), len(reference_line_list[j][reference_line_valid_mask_list[j]]))
                    diff = np.abs(
                        reference_line_list[i][reference_line_valid_mask_list[i]][:min_len, :2] - reference_line_list[j][reference_line_valid_mask_list[j]][:min_len, :2]
                    ).sum(-1)
                    if np.max(diff) < 0.5:
                        remove_index.add(j)
                        
            reference_line_list = [
                reference_line_list[i] for i in range(len(reference_line_list)) if i not in remove_index
            ]
            
            n_points = int(120 / 1.0)
            position = np.zeros((len(reference_line_list), n_points, 2), dtype=np.float64)
            vector = np.zeros((len(reference_line_list), n_points, 2), dtype=np.float64)
            orientation = np.zeros((len(reference_line_list), n_points), dtype=np.float64)
            valid_mask = np.zeros((len(reference_line_list), n_points), dtype=np.bool_)
            future_projection = np.zeros((len(reference_line_list), 8, 2), dtype=np.float64)
            
            ego_future = state.log_trajectory.xy[de, bs, :, 11:]
            is_sdc_idx = state.object_metadata.is_sdc[de, bs]
            ego_future_valid = state.log_trajectory.valid[de, bs, :, 11:][is_sdc_idx]
            ego_future = ego_future[is_sdc_idx][ego_future_valid]
            av_curr_xy = state.current_log_trajectory.xy[de, bs][is_sdc_idx].squeeze(0)
            av_curr_yaw = state.current_log_trajectory.yaw[de, bs][is_sdc_idx].squeeze(0)
            rotate_mat = np.array(
                [
                    [np.cos(av_curr_yaw), -np.sin(av_curr_yaw)],
                    [np.sin(av_curr_yaw), np.cos(av_curr_yaw)],
                ],
                dtype=np.float64,
            ).squeeze(-1)
            if ego_future.shape[0] > 0:
                linestring = [
                    LineString(reference_line_list[i]) for i in range(len(reference_line_list))
                ]
                future_samples = ego_future[9::10]  # every 1s
                future_samples = [Point(xy) for xy in future_samples]
            
            for i, line in enumerate(reference_line_list):
                subsample = line[reference_line_valid_mask_list[i]]
                n_valid = len(subsample)
                # try:
                position[i, : n_valid - 1] = subsample[:-1, :2]
                # except:
                #     print('hrer')
                vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)
                orientation[i, : n_valid - 1] = subsample[:-1, 2]
                valid_mask[i, : n_valid - 1] = True
                
                if ego_future.shape[0] > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for j, future_sample in enumerate(future_samples):
                            future_projection[i, j, 0] = linestring[i].project(
                                future_sample
                            )
                            future_projection[i, j, 1] = linestring[i].distance(
                                future_sample
                            )

            position = np.matmul(position - av_curr_xy, rotate_mat)
            vector = np.matmul(
                vector, rotate_mat
            )
            orientation -= np.array(av_curr_yaw)
            
            # import matplotlib.pyplot as plt
            # for bat in range(26):
            #     plt.plot(position[bat][:, 0][valid_mask[bat]], position[bat][:, 1][valid_mask[bat]], 'r-')
            #     plt.plot(position[bat][0, 0][valid_mask[bat, 0]], position[bat][0, 1][valid_mask[bat, 0]], 'bo')
            # plt.savefig(f"/home/jyyun/workshop/LatentDriver/vis/{bat}_ref.png")
            # plt.close()

            b_position_list.append(to_tensor(position))
            b_vector_list.append(to_tensor(vector))
            b_orientation.append(to_tensor(orientation))
            b_valid_mask.append(to_tensor(valid_mask))
            b_future_projection.append(to_tensor(future_projection))

        d_position_list.append(b_position_list)
        d_vector_list.append(b_vector_list)
        d_orientation.append(b_orientation)
        d_valid_mask.append(b_valid_mask)
        d_future_projection.append(b_future_projection)
    
    return {"position": d_position_list,
            "vector": d_vector_list,
            "orientation": d_orientation,
            "valid_mask": d_valid_mask,
            "future_projection": d_future_projection
    }
    
def get_obs(*args):
    data_dict,sdc_obs = get_obs_from_routeandmap_saved(*args)
    obs = preprocess_data_dist_jnp(data_dict)
    return obs,data_dict

class WaymoEnv():
    def __init__(self,
                waymax_conf,
                data_conf,
                env_conf,
                batch_dims,
                ego_control_setting:dict,
                metric_conf:dict,):
        '''
        ego_control_setting:
            ego_policy_type:
                'expert'
                'idm': not implemented
                'custom'
            action_type: bicycle or waypoints
            action_space = dict(
                type = 'continuous', # can be 'continuous' or 'discrete',
                action_ranges = action_ranges,
                bins = [13,39] #for acc and steer only for discrete
            )
            npc_policy_type:
                'expert'
                'idm'
        metric_conf:
            arrival_thres = 0.9
            intention_label_path =/root/intention_label_val

        '''
        self.ego_policy_type = ego_control_setting.ego_policy_type
        action_type = ego_control_setting.action_type
        action_space = ego_control_setting.action_space
        npc_type = ego_control_setting.npc_policy_type
        self.npc_type = npc_type
        self.data_conf = data_conf
        self.num_devices = batch_dims[0]
        multi_device = self.num_devices > 1
        if action_type == 'bicycle':
            env_dynamic_model = dynamics.InvertibleBicycleModel()
        elif action_type == 'waypoint':
            env_dynamic_model = dynamics.DeltaLocal()

        if action_space.type == 'continuous':
            pass
        elif action_space.type == 'discrete':
            raise Warning('discrete action space has not been verified!')
            self.discretizer = build_discretizer(action_space)
        self.action_space_type = action_space.type
        self._catk_multi = (npc_type == 'catk') and multi_device
        if npc_type == 'catk':
            self.com_traj = combin_traj
            self.dynamic_inverse = env_dynamic_model.inverse
            self.get_obs_fn = get_obs_from_routeandmap_saved_jit
            self._obs_uses_pmap = False
        else:
            if multi_device:
                self.com_traj = jax.pmap(combin_traj)
                self.dynamic_inverse = jax.pmap(env_dynamic_model.inverse)
                self.get_obs_fn = get_obs_from_routeandmap_saved_pmap
                self._obs_uses_pmap = True
            else:
                self.com_traj = jax.jit(combin_traj)
                self.dynamic_inverse = jax.jit(env_dynamic_model.inverse)
                self.get_obs_fn = get_obs_from_routeandmap_saved_jit
                self._obs_uses_pmap = False
        self.env = WaymoBaseEnv(
                    waymax_conf=waymax_conf,
                    env_conf=env_conf,
                    action_space=action_space,
                    action_type=action_type,
                    dynamics_model=env_dynamic_model,
                    npc = npc_type,
                    num_devices=self.num_devices)

        self.metric = Metric(**metric_conf, batch_dims=batch_dims)
        self.log_rew_dict = {}
        self.batch_dims = batch_dims
        self.states = []
        self.state = None

        self.path_to_map = os.path.join(data_conf.path_to_processed_map_route,'map')
        self.path_to_route = os.path.join(data_conf.path_to_processed_map_route,'route')

    def _compute_obs(self, state):
        if self._catk_multi:
            return self._compute_obs_catk_multi(state)
        if self._obs_uses_pmap:
            state_for_obs = state
            # pmap requires the first dimension to match num_devices
            # Replicate map and route data across devices
            map_arg = self.road_np
            route_arg = self.route_np
        else:
            if self.num_devices == 1:
                squeeze_leading_axis = lambda arr: arr[0] if (hasattr(arr, "shape") and arr.shape != () and arr.shape[0] == 1) else arr
                state_for_obs = jax.tree_util.tree_map(squeeze_leading_axis, state)
                map_arg = jax.device_put(self.road_np[0])
                route_arg = jax.device_put(self.route_np[0])
            else:
                state_for_obs = state
                map_arg = jax.device_put(self.road_np)
                route_arg = jax.device_put(self.route_np)
        data_dict, _ = self.get_obs_fn(state_for_obs, map_arg, route_arg, (80, 20))
        data_dict = jax.tree_util.tree_map(jax.device_get, data_dict)
        if not self._obs_uses_pmap and self.num_devices == 1:
            data_dict = {k: v[np.newaxis, ...] for k, v in data_dict.items()}
        obs = preprocess_data_dist_jnp(data_dict)
        return obs, data_dict

    def _compute_obs_catk_multi(self, state):
        state_host = jax.tree_util.tree_map(
            lambda arr: jax.device_get(arr) if hasattr(arr, "shape") else arr,
            state,
        )

        def slice_first_axis(arr, idx):
            if not hasattr(arr, "shape"):
                return arr
            if arr.shape == ():
                return arr
            if arr.shape[0] == self.num_devices:
                return arr[idx]
            return arr

        data_list = []
        for dev_idx in range(self.num_devices):
            state_i = jax.tree_util.tree_map(lambda x, i=dev_idx: slice_first_axis(x, i), state_host)
            if getattr(state_i, "shape", ()) == ():
                state_i = jax.tree_util.tree_map(
                    lambda arr: np.expand_dims(arr, axis=0) if hasattr(arr, "shape") and arr.shape != () else arr,
                    state_i,
                )
            map_i = slice_first_axis(self.road_np, dev_idx)
            route_i = slice_first_axis(self.route_np, dev_idx)
            data_i, _ = get_obs_from_routeandmap_saved_jit(state_i, map_i, route_i, (80, 20))
            data_list.append(jax.tree_util.tree_map(lambda arr: jax.device_get(arr), data_i))

        data_dict = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *data_list)
        obs = preprocess_data_dist_jnp(data_dict)
        return obs, data_dict

    def get_expert_action(self)->np.ndarray:
        current_state = self.states[-1]
        traj = self.com_traj(current_state)
        action = self.dynamic_inverse(traj, current_state.object_metadata,jnp.zeros(self.batch_dims[0],dtype=jnp.int32))
        action = np.array(action.data[current_state.object_metadata.is_sdc])
        return action

    def reset(self):
        self.scenario = next(self.env.data_iter)
        initial_state = self.env.pmap_reset(self.scenario)
        self.states = [initial_state]
        cur_state = initial_state
        self.road_np, self.route_np, self.intention_label = get_cache_polylines_baseline(cur_state, self.path_to_map, self.path_to_route, self.metric.intention_label_path)
        self.metric.reset(self.intention_label)
        obs, obs_dict = self._compute_obs(cur_state)
        
        reference_lines = get_reference_line(cur_state)
        target = None
        # reference_lines, target = None, None
        return obs, obs_dict, reference_lines, target

    def step(self,action=None, show_global=False):
        # check ego agent control mode
        if self.ego_policy_type == 'custom':
            pass
        elif self.ego_policy_type == 'stationary':
            action = np.zeros_like(action)
        elif self.ego_policy_type == 'expert':
            action = self.get_expert_action()
        else:
            raise ValueError(f'ego_policy_type {self.ego_policy_type} not supported, only support expert, stationary and custom')

        current_state = self.states[-1]
        info = {}
        if self.action_space_type =='discrete':
            action = self.discretizer.make_continuous(action)
            # print(action)
        # (N,B,action_space)
        action = action.reshape(self.batch_dims[0],self.batch_dims[1], -1)
        # (N,action_space,B)
        actions = np.transpose(action,(0,2,1))


        rewards,rew,next_state = self.env.pmap_sim(actions,current_state)
        obs, obs_dict = self._compute_obs(next_state)
        is_done = np.asarray(jax.device_get(next_state.is_done))
        is_done = is_done.reshape(-1)
        
        reference_lines = get_reference_line(next_state)
        # reference_lines = None
        done = np.repeat(is_done, self.batch_dims[-1]).astype(bool)
        self.states.append(next_state)
        self.metric.update(rewards,rew)
        # logger
        for k,_ in rewards.items():
            if k not in self.log_rew_dict:
                self.log_rew_dict[k] = rewards[k]
            else:
                self.log_rew_dict[k] += rewards[k]

        if done[-1]:
            # logger
            for k,v in self.log_rew_dict.items():
                info['reward/'+k] = v.mean()
            # metric
            self.metric.collect_batch(info)
            if show_global:
                print('\n',self.metric.get_global_info())
        return (obs, obs_dict,np.array(rew).reshape(self.num_envs,), done, info, reference_lines)


    @property
    def observation_space(self) -> gym.spaces.Space:
        raise NotImplementedError
        # return gym.spaces.Box(low= -float('inf'), high=float('inf'), shape=(self.cfg.obs_shape), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space_

    @property
    def reward_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(1, ), dtype=np.float32
            )
    @property
    def num_envs(self)->int:
        return self.batch_dims[0] * self.batch_dims[1]

    def get_env_idx(self,batch_id=None):
        id_bank = np.array(self.states[-1]._scenario_id.reshape(-1), dtype=np.uint64).tolist()
        if batch_id == None:
            return id_bank
        else:
            return id_bank[batch_id]
