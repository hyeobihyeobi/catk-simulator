import sys

import numpy as np
import gymnasium as gym
import numpy as np
import waymax
import jax
import jax.numpy as jnp
from waymax import config as _config
from waymax import dynamics
from waymax import env as _env
from waymax import datatypes
from waymax import agents
from waymax import dataloader
from waymax.config import DatasetConfig
from simulator.actor import create_control_actor
from src.dataloader.waymax_scenloader import WaymoScenLoader
class WaymoBaseEnv():
    def __init__(self,
                waymax_conf,
                env_conf,
                dynamics_model,
                action_space,
                action_type,
                npc:str,
                num_devices:int=1) -> None:
        is_customized = waymax_conf.pop('customized')
        if not is_customized:
            self.data_iter = dataloader.simulator_state_generator(config=DatasetConfig(**waymax_conf))
        elif is_customized:
            raise NotImplementedError("Customized dataset is not supported yet")
            self.data_iter = WaymoScenLoader(waymax_conf)

        if action_type == 'waypoint':
            # for dx dy dyaw
            shape = (3,)
            # ranges = [(-0.14, 6), (-0.35, 0.35), (-0.15,0.15)]
            ranges = action_space.action_ranges 
            low = np.array([r[0] for r in ranges])
            high = np.array([r[1] for r in ranges])
            self.action_space_ = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
        elif action_type == 'bicycle':
            # for acc steers
            shape = (2,)
            ranges = action_space.action_ranges #orignal -1
            low = np.array([r[0] for r in ranges])
            high = np.array([r[1] for r in ranges])
            self.action_space_ = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
        else:
            raise ValueError("action_type should be waypoint or bicycle")

        self.dynamics_model = dynamics_model
        # create actors
        actor = create_control_actor(is_controlled_func = lambda state: state.object_metadata.is_sdc)
        actors = [actor]
        npc_policy_type = npc
        if npc_policy_type=='idm':
            controlled_object = _config.ObjectType.VALID
            npc_actor = agents.IDMRoutePolicy(
            is_controlled_func=lambda state: ~state.object_metadata.is_sdc,
            # additional_lookahead_points = 40,
            # additional_lookahead_distance = 40.0,
            )
            actors.append(npc_actor)
        elif npc_policy_type=='catk':
            controlled_object = _config.ObjectType.VALID
            # Create multiple CATK agents, one per GPU
            num_devices = jax.local_device_count()
            self.catk_agents = []
            for device_id in range(num_devices):
                npc_actor = agents.CATK_Simulator(
                    is_controlled_func=lambda state: ~state.object_metadata.is_sdc,
                    device_id=device_id,
                )
                self.catk_agents.append(npc_actor)
                if device_id == 0:  # Only add first one to actors list for compatibility
                    actors.append(npc_actor)
        elif npc_policy_type=='expert':
            controlled_object = _config.ObjectType.SDC
        else:
            raise ValueError("npc should be idm or expert")
        env_conf.update({'controlled_object': controlled_object})
        self.select_action_list = [actor.select_action for actor in actors]

        self.env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics.StateDynamics(),
        config=_config.EnvironmentConfig(**env_conf)
        )
        if npc_policy_type=='catk':
            if num_devices > 1:
                self.pmap_sim = self.simulate_catk_multi_gpu
                self.pmap_reset = jax.pmap(self.reset_func)
                sys.stdout.write(f'\n\n Using CATK NPC policy with {num_devices} GPUs \n\n')
            else:
                self.pmap_sim = self.simulate
                self.pmap_reset = self.reset_func
                sys.stdout.write('\n\n Using CATK NPC policy (single GPU) \n\n')
            sys.stdout.flush()
        else:
            self.pmap_sim = jax.pmap(self.simulate)
            self.pmap_reset = jax.pmap(self.reset_func)

    def reset_func(self,scenario: datatypes.SimulatorState):
        """Reset a single scenario and return a single state.

        This function is pmapped over devices. It must NOT return a Python
        list; it should return the state object directly so that pmap can
        stack them along the device axis.
        """
        return self.env.reset(scenario)
    def simulate_catk_multi_gpu(self, actions: np.array, current_state: datatypes.SimulatorState):
        """Wrapper for CATK simulation with multi-GPU support."""
        num_devices = len(self.catk_agents)

        # Process each device batch separately with its corresponding GPU
        results = []
        for device_id in range(num_devices):
            # Extract the slice for this device (already split by pmap)
            state_slice = jax.tree_map(lambda x: x[device_id] if (hasattr(x, "__getitem__") and hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == num_devices) else x, current_state)
            actions_slice = actions[device_id] if actions.shape[0] == num_devices else actions

            # Temporarily replace the CATK agent in select_action_list
            old_select_action = self.select_action_list[1] if len(self.select_action_list) > 1 else None
            if old_select_action:
                self.select_action_list[1] = self.catk_agents[device_id].select_action

            # Run simulation for this device
            result = self.simulate(actions_slice, state_slice)
            results.append(result)

            # Restore original select_action
            if old_select_action:
                self.select_action_list[1] = old_select_action

        # Stack results back into device dimension
        merged_rewards = jax.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *[r[0] for r in results]
        )
        merged_rew = jnp.stack([r[1] for r in results], axis=0)
        merged_state = jax.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *[r[2] for r in results]
        )

        return merged_rewards, merged_rew, merged_state

    def simulate(self, actions:np.array, current_state: datatypes.SimulatorState):
        outputs = [
            select_action({'actions':actions}, current_state, None, None)
            for select_action in self.select_action_list
        ]
        ego_output = outputs[0]
        ego_action = ego_output.action
        data = ego_action.data
        if data.ndim == 4 and data.shape[0] == 3:
            data = jnp.moveaxis(data, 0, -1)[..., 0, :]  # (32, 128, 3)
            data = data.astype(jnp.float32)[jnp.newaxis, ...]  # (1, 32, 128, 3)
            valid = ego_action.valid
            target_valid_shape = data[..., :1].shape
            if valid.ndim == 4 and valid.shape[0] == 3:
                valid = jnp.moveaxis(valid, 0, -1)[..., 0, :]
                valid = valid[jnp.newaxis, ...]
            if valid.shape != target_valid_shape:
                valid = jnp.ones(target_valid_shape, dtype=bool)
            ego_action = datatypes.Action(data=data, valid=valid)
            outputs[0] = ego_output.__class__(
                actor_state=ego_output.actor_state,
                action=ego_action,
                is_controlled=ego_output.is_controlled,
            )
        traj = datatypes.dynamic_slice(
                    inputs=current_state.sim_trajectory, start_index=current_state.timestep.flatten()[0], slice_size=1, axis=-1
                )
        if not hasattr(self, "_action_shape_logged"):
#             print("[WaymoBaseEnv] control action shape:", outputs[0].action.data.shape)
#             print("[WaymoBaseEnv] npc action shape:", outputs[1].action.data.shape if len(outputs) > 1 else None)
            self._action_shape_logged = True
        action_transformed = self.dynamics_model.compute_update(outputs[0].action, traj).as_action()
        outputs[0].action.data = action_transformed.data.astype(jnp.float32)
        # outputs[0].action.valid = action_transformed.valid
        # jax.debug.print("sdc: {x} \n , anothers: {y}",x = outputs[0].action.valid, y=outputs[1].action.valid)
        action = waymax.agents.merge_actions(outputs)
        reward = self.env.reward(current_state, action)
        rewards,rew =datatypes.select_by_onehot(
            reward, current_state.object_metadata.is_sdc, keepdims=False
        )
        next_state = self.env.step(current_state, action)
        return rewards,rew,next_state
