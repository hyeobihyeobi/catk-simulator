from simulator.engines.base_simulator import BaseSimulator
import time
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

class LTDSimulator(BaseSimulator):
    def __init__(self, model, config, batch_dims):
        super().__init__(model, config, batch_dims)
        self.cfg = config
    
    def run(self,ep_return:list, vis):
        self.idx = 0
        while True:
            try:
                obs, obs_dict, reference_lines, target = self.env.reset()
                
                # import pdb; pdb.set_trace()
                reference_lines = {
                    k: pad_sequence(
                        [f.to(self.device) for j in range(len(reference_lines[k])) for f in reference_lines[k][j]], batch_first=True
                    )
                    for k in reference_lines.keys()
                }
                
                # import matplotlib.pyplot as plt
                # for li, lane in enumerate(reference_lines["position"][2]):
                #     plt.plot(lane[:, 0][reference_lines["valid_mask"][2][li]].cpu(), lane[:, 1][reference_lines["valid_mask"][2][li]].cpu(), 'g-')
                # plt.savefig("/home/jyyun/workshop/LatentDriver/vis/temp.png")
                # plt.close()
                
                # import pdb; pdb.set_trace()
                obs = obs.reshape(self.env.num_envs,-1,7) #(B, N, 7)
                obs_depth,obs_dim = obs.shape[1],obs.shape[-1]
                states = (obs).reshape(self.env.num_envs,-1,obs_depth,obs_dim)
                target_return = torch.tensor(ep_return, device=self.device, dtype=torch.float32).reshape(
                    self.env.num_envs, -1, 1
                )
                timesteps = torch.tensor([0] * self.env.num_envs, device=self.device, dtype=torch.long).reshape(
                    self.env.num_envs, -1
                ) #(B, 1) #0값으로 채워짐
                if self.cfg.action_space.dynamic_type == 'bicycle':
                    actions = np.zeros((self.env.num_envs, 1,2))
                else:
                    actions = np.zeros((self.env.num_envs, 1,3))
                rewards = np.zeros((self.env.num_envs, 1,1))
                done_ = False
                self.T = 1
                a = time.time()
                

                while not done_:
                    rewards = np.concatenate(
                        [
                            rewards,
                            np.zeros((self.env.num_envs, 1)).reshape(self.env.num_envs, -1, 1),
                        ],
                        axis=1,
                    ) #(B, 2, 1)
                    with torch.no_grad():
                        action = self.model.get_predictions(
                            torch.tensor(states,device =self.device),
                            torch.tensor(actions,device =self.device),
                            reference_lines,
                            timesteps.to(dtype=torch.long),
                            num_envs=self.env.num_envs
                        )
                    
                        
                    if isinstance(action,torch.Tensor):
                        action = action.detach().cpu().numpy()
                    control_action = action

                    obs, obs_dict,rew, done, info, reference_lines = self.env.step(control_action,show_global=True)
                    
                    reference_lines = {
                        k: pad_sequence(
                            [f.to(self.device) for j in range(len(reference_lines[k])) for f in reference_lines[k][j]], batch_first=True
                        )
                        for k in reference_lines.keys()
                    }
                    
                    actions = np.concatenate([actions,action[:,np.newaxis,...]],axis=1)
                    # actions[:, -1] = action
                    obs = obs.reshape(self.env.num_envs,-1,7)
                    state = (obs.reshape(self.env.num_envs,-1,obs_depth,obs_dim))
                    states = np.concatenate([states,state],axis=1)
                    reward = rew.reshape(self.env.num_envs,1)
                    rewards[:,-1] = reward

                    pred_return = target_return[:, -1]-(torch.tensor(reward,device=self.device))
                    target_return = torch.cat(
                        [target_return, pred_return.reshape(self.env.num_envs, -1, 1)], dim=1
                    )

                    timesteps = torch.cat(
                        [
                            timesteps,
                            torch.ones((self.env.num_envs, 1), device=self.device, dtype=torch.long).reshape(
                                self.env.num_envs, 1
                            )
                            * (self.T),
                        ],
                        dim=1,
                        )

                    self.T+=1
                    done_ =done[-1]
                self.idx += 1
                print('Processed: ', self.idx, 'th batch, Time: ', time.time()-a, 's')

                self.render(vis, self.cfg.method.model_name)

            except StopIteration:
                print("StopIteration")
                break