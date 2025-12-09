import src.utils.init_default_jax
import jax
import hydra
import os
from omegaconf import OmegaConf
from src.utils.utils import update_waymax_config, saving_data
import time
from simulator.waymo_env import WaymoEnv
import numpy as np
from waymax import dynamics
import jax.numpy as jnp
import requests

class TrainingDataCollector():
    def __init__(self,
                config,
                ):
        self.env = WaymoEnv(
            waymax_conf=config.waymax_conf,
            env_conf=config.env_conf,
            batch_dims=config.batch_dims,
            ego_control_setting=config.ego_control_setting,
            metric_conf=config.metric_conf,
            data_conf=config.data_conf,
        )
        self.save_path = config.save_path
        self.batch_dims = config.batch_dims
        self.size = self.batch_dims[0] * self.batch_dims[1]
        dynamics_model_bicycle = dynamics.InvertibleBicycleModel()
        dynamics_model_waypoints = dynamics.DeltaLocal()
        self.get_action_bicycle = jax.pmap(dynamics_model_bicycle.inverse)
        self.get_action_waypoints = jax.pmap(dynamics_model_waypoints.inverse)

    def _merge_devices(self, arr):
        """Merge leading device/batch dims into a single env axis and tile if needed."""
        arr = np.array(arr)
        # If only one route per device, tile across batch dim to align with envs.
        if (
            arr.ndim >= 2
            and arr.shape[0] == self.batch_dims[0]
            and arr.shape[1] == 1
            and self.batch_dims[1] > 1
        ):
            tile_shape = (1, self.batch_dims[1]) + (1,) * (arr.ndim - 2)
            arr = np.tile(arr, tile_shape)
        if arr.ndim >= 3 and arr.shape[0] * arr.shape[1] == self.env.num_envs:
            return arr.reshape(self.env.num_envs, *arr.shape[2:])
        if arr.ndim >= 2 and arr.shape[0] == self.env.num_envs:
            return arr
        if arr.ndim >= 1 and arr.shape[0] == 1 and self.env.num_envs > 1:
            reps = (self.env.num_envs,) + (1,) * (arr.ndim - 1)
            return np.tile(arr, reps)
        return arr

    def _format_obs(self, obs_dict):
        """Convert obs dict from WaymoEnv into env-major numpy arrays."""
        formatted = {k: self._merge_devices(v) for k, v in obs_dict.items()}
        return formatted

    def _format_reference_lines(self, reference_lines):
        """Flatten reference line outputs (device, batch) -> (env, L, ...), padded to max L."""
        if reference_lines is None or len(reference_lines) == 0:
            return {}
        keys = ["position", "vector", "orientation", "valid_mask", "future_projection"]
        flat = {k: [] for k in keys}
        max_lines = 0
        num_envs = self.env.num_envs
        for dev_idx, dev_list in enumerate(reference_lines.get("position", [])):
            for batch_idx in range(len(dev_list)):
                for k in keys:
                    item = reference_lines[k][dev_idx][batch_idx]
                    if hasattr(item, "detach"):
                        item = item.detach().cpu().numpy()
                    else:
                        item = np.array(item)
                    flat[k].append(item)
                    max_lines = max(max_lines, item.shape[0] if item.ndim > 0 else 0)

        formatted = {}
        for k, items in flat.items():
            if not items:
                formatted[k] = np.zeros((num_envs, 0))
                continue
            padded_items = []
            for arr in items:
                if arr.shape[0] < max_lines:
                    pad_shape = (max_lines - arr.shape[0],) + arr.shape[1:]
                    arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
                padded_items.append(arr)
            formatted[k] = np.stack(padded_items, axis=0)
        # If some envs had no ref line, pad with zeros.
        for k, arr in formatted.items():
            if arr.shape[0] < num_envs:
                pad_shape = (num_envs - arr.shape[0],) + arr.shape[1:]
                formatted[k] = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
        return formatted

    def run(self):
        self.idx = 0
        while True:
            try:
                a = time.time()
                obs, obs_dict, reference_lines, target = self.env.reset()
                # import pdb; pdb.set_trace()
                if obs == None:
                    continue
                
                obs_formatted = self._format_obs(obs)
                # ref_formatted = self._format_reference_lines(reference_lines)

                # obs_seq = {k: [v] for k, v in obs_formatted.items()}
                # # ref_seq = {k: [v] for k, v in ref_formatted.items()}
                # actions_bicycle = []
                # actions_waypoints = []
                # rewards = []
                # done_ = False

                sdc_gt_raw = obs_dict.get('sdc_gt_traj', None)
                agent_gt_raw = obs_dict.get('agent_gt_traj', None)
                sdc_gt_formatted = self._merge_devices(sdc_gt_raw) if sdc_gt_raw is not None else None
                agent_gt_formatted = self._merge_devices(agent_gt_raw) if agent_gt_raw is not None else None

#                 while not done_:
#                     actions_to_collect = self.collect_actions(self.env.states[-1])
#                     actions_bicycle.append(actions_to_collect['bicycle_actions'])
#                     actions_waypoints.append(actions_to_collect['waypoints_actions'])
# 
#                     obs, obs_dict, rew, done, info, reference_lines = self.env.step(
#                         self.env.get_expert_action(), show_global=False
#                     )
#                     rewards.append(rew.reshape(self.env.num_envs, 1))
# 
#                     obs_formatted = self._format_obs(obs)
#                     for k in obs_seq:
#                         obs_seq[k].append(obs_formatted[k])
# 
#                     # ref_formatted = self._format_reference_lines(reference_lines)
#                     # for k in ref_seq:
#                     #     ref_seq[k].append(ref_formatted.get(k, np.zeros_like(ref_seq[k][0])))
# 
#                     done_ = done[-1]
# 
#                 obs_stacked = {k: np.stack(v, axis=1) for k, v in obs_seq.items()}
#                 # ref_stacked = {k: np.stack(v, axis=1) for k, v in ref_seq.items()} if ref_seq else {}
#                 actions_bicycle = (
#                     np.stack(actions_bicycle, axis=1) if actions_bicycle else np.zeros((self.env.num_envs, 0, 2))
#                 )
#                 actions_waypoints = (
#                     np.stack(actions_waypoints, axis=1) if actions_waypoints else np.zeros((self.env.num_envs, 0, 3))
#                 )
#                 rewards = np.stack(rewards, axis=1) if rewards else np.zeros((self.env.num_envs, 0, 1))
# 
#                 time_horizon = next(iter(obs_stacked.values())).shape[1]
#                 terminals = np.zeros(time_horizon, dtype=np.int32)
#                 terminals[-1] = 1
# 
#                 for ii in range(self.env.num_envs):
#                     scen_id = str(self.env.get_env_idx(ii))
#                     sub_folder = os.path.join(self.save_path, 'data')
#                     os.makedirs(sub_folder, exist_ok=True)
#                     traj = {
#                         'obs': {k: v[ii] for k, v in obs_stacked.items()},
#                         # 'reference_lines': {k: v[ii] for k, v in ref_stacked.items()},
#                         'waypoints_actions': actions_waypoints[ii],
#                         'bicycle_actions': actions_bicycle[ii],
#                         'rewards': rewards[ii],
#                         'terminals': terminals,
#                     }
#                     saving_data(traj, name=os.path.join(sub_folder, scen_id))
#                     with open(os.path.join(self.save_path, 'name.txt'), 'a') as f:
#                         f.write(f'{scen_id}\n')

                sub_folder = os.path.join(self.save_path, 'data')
                os.makedirs(sub_folder, exist_ok=True)
                for ii in range(self.env.num_envs):
                    scen_id = str(self.env.get_env_idx(ii))
                    traj = {
                        'obs': {k: v[ii] for k, v in obs_formatted.items()},
                        'sdc_gt': None if sdc_gt_formatted is None else sdc_gt_formatted[ii],
                        'agent_gt': None if agent_gt_formatted is None else agent_gt_formatted[ii],
                    }
                    saving_data(traj, name=os.path.join(sub_folder, scen_id))
                    with open(os.path.join(self.save_path, 'name.txt'), 'a') as f:
                        f.write(f'{scen_id}\n')



                self.idx += 1
                print('Processed: ', self.idx, 'th batch, Time: ', time.time()-a, 's')

            except StopIteration:
                print("StopIteration")
                break

    def collect_actions(self,next_state):
        action_collected = {}
        traj = self.env.com_traj(next_state)
        '''for bicycles'''
        action = self.get_action_bicycle(traj,metadata=next_state.object_metadata, timestep=jnp.zeros(self.batch_dims[0],dtype=jnp.int32))
        action = np.array(action.data[next_state.object_metadata.is_sdc]).reshape(self.env.num_envs, -1)

        # make acc steer here become (B,1)
        # acc,steer = action[...,0:1],action[...,1:2]
        action_collected.update(bicycle_actions = action)
        '''for waypoints'''
        action = self.get_action_waypoints(traj,metadata=next_state.object_metadata, timestep=jnp.zeros(self.batch_dims[0],dtype=jnp.int32))
        action = np.array(action.data[next_state.object_metadata.is_sdc]).reshape(self.env.num_envs, -1)
        # dx,dy,dyaw = action[...,0:1],action[...,1:2],action[...,2:3]
        action_collected.update(waypoints_actions = action)
        return action_collected

@hydra.main(version_base=None, config_path="../../configs", config_name="simulate")
def run(cfg):
    """
    Entry point for the data collection script.

    Args:
        cfg (OmegaConf): The configuration object.
    """
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = update_waymax_config(cfg)
    cfg = OmegaConf.merge(cfg, cfg.method)
    collector = TrainingDataCollector(cfg)
    collector.run()

if __name__ == '__main__':
    message="SNU Ubuntu :\nSomething Went Wrong! (code exit with error)".encode(encoding='utf-8')
    try:
        run()
        message="SNU Ubuntu :\nPreprocessing done successful".encode(encoding='utf-8')
    except Exception as e:
        message=f"SNU Ubuntu :\nPreprocessing failed with error: {e}".encode(encoding='utf-8')
        raise e
    finally:
        requests.post("https://ntfy.sh/shnamtopic", data=message)
