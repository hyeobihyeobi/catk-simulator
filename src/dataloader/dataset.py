from torch.utils.data import Dataset
from src.utils.utils import loading_data
import os
import torch
import random
import numpy as np
from src.dataloader.normalizer import Normalizer
MAX_EPISODE_LEN = 80
def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret
class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        act_key,
        reward_scale,
        action_range,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = 7
        self.reward_scale = reward_scale
        self.normalizer = Normalizer(action_range)
        # the user defined action range.
        self.action_range = action_range
        if act_key == 'bicycle':
            self.act_key = 'bicycle_actions'
            self.act_dim = 2
        elif act_key == 'waypoint':
            self.act_key = 'waypoints_actions'
            self.act_dim = 3
    def __call__(self, traj, si):
        """
        Adapted for new preprocessed data:
        traj = {'obs': <dict of numpy arrays>, 'sdc_gt': ..., 'agent_gt': ...}
        We simply return the observation dict and GT trajectories.
        """
        obs = traj.get("obs", {})
        sdc_gt = traj.get("sdc_gt", None)
        agent_gt = traj.get("agent_gt", None)
        # Convert to torch tensors where possible for downstream code.
        obs_torch = {}
        for k, v in obs.items():
            obs_torch[k] = v if torch.is_tensor(v) else torch.tensor(v)
        sdc_gt_torch = None if sdc_gt is None else (sdc_gt if torch.is_tensor(sdc_gt) else torch.tensor(sdc_gt))
        agent_gt_torch = None if agent_gt is None else (agent_gt if torch.is_tensor(agent_gt) else torch.tensor(agent_gt))
        return obs_torch, sdc_gt_torch, agent_gt_torch
        # return dict(
        #     states=ss,
        #     actions=aa,
        #     actions_gt = aa_gt,
        #     rewards=rr,
        #     dones=dd,
        #     rtg=rtg,
        #     timesteps=timesteps,
        #     ordering=ordering,
        #     padding_mask=padding_mask,
        # )

class WaymoDataLoader(Dataset):
    def __init__(self,config) -> None:
        self.dir = config.data_path
        self.full_name_list = []
        with open(os.path.join(self.dir,'name.txt')) as f:
            for name in f.readlines():
                self.full_name_list.append(name.strip())
        if config.mini == True:
            random.shuffle(self.full_name_list)
            self.full_name_list = self.full_name_list[:int(len(self.full_name_list)*0.1)]
            print("Using random 0.1 for trainig")
        start_t_idx = np.arange(0, MAX_EPISODE_LEN - config.max_len + 1).tolist()
        if config.overlap_sample==True:
            raise NotImplementedError('Not implemented yet for overlap_sample')
        else:
            # e.g. max_len is 10, MAX_EPISODE_LEN is 80
            # start_t_idx = [0 ... 70] -> [0, 10, 20, 30, 40, 50, 60, 70], every data has no overlap
            start_t_idx = start_t_idx[::config.max_len]
        aug_list = []
        print('Preparing split...')
        for name in self.full_name_list:
            for start_idx in start_t_idx:
                aug_list.append(f"{name}-{start_idx}")
        self.full_name_list = aug_list.copy()
        del aug_list
        self.transform = TransformSamplingSubTraj(
            max_len=config.max_len,
            act_key=config.action_space.dynamic_type,
            reward_scale=1,
            action_range=config.action_space.action_ranges
        )
    def __len__(self):
        return len(self.full_name_list)
    def __getitem__(self, index):
        name, si = self.full_name_list[index].split('-')[0],self.full_name_list[index].split('-')[1]
        traj = loading_data(os.path.join(self.dir,'data',name))
        return self.transform(traj,int(si))
        # return state,action

import hydra
@hydra.main(version_base=None, config_path="../configs", config_name="train")
def debug(cfg):
    from omegaconf import OmegaConf
    from tqdm import tqdm
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    loader = WaymoDataLoader(config=cfg)
    for idx in tqdm(range(len(loader))):
        # if loader.full_name_list[idx] == '618217707':
        ss, aa, aa_gt, rr, dd, rtg, timesteps, ordering, padding_mask = loader.__getitem__(idx)

if __name__ == '__main__':
    debug()
