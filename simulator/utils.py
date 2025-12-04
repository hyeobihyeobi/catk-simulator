import numpy as np
import jax.numpy as jnp
import jax
from waymax import datatypes
from src.utils.utils import loading_data
from src.utils.discretizer import Discretizer
import os

_MISSING_ROUTE_IDS = set()
_MISSING_LABEL_IDS = set()
_ROUTE_FALLBACK_TEMPLATE = np.zeros((20, 6), dtype=np.float64)
_TL_FALLBACK_TEMPLATE = np.zeros((0, 6), dtype=np.float64)
_ON_ROUTE_FALLBACK_TEMPLATE = np.zeros((20, 1), dtype=np.bool_)

def get_cache_polylines_baseline(cur_state:datatypes.SimulatorState,
                                path_to_map:str,
                                path_to_route:str,
                                intention_label_path:str=None):

    cur_id = cur_state._scenario_id.reshape(cur_state.shape)
    # cur_id may be (num_devices,) or (num_devices, B)
    whole_map_by_device_id = []
    whole_route_by_device_id = []
    intention_label = []
    dev_dim = cur_id.shape[0] if hasattr(cur_id, 'shape') else 1
    has_batch = (hasattr(cur_id, 'ndim') and cur_id.ndim >= 2)
    for device_id in range(dev_dim):
        whole_map_by_batch = []
        whole_route_by_batch = []
        batch_range = range(cur_id.shape[1]) if has_batch else range(1)
        for batch_id in batch_range:
            scenario_id = cur_id[device_id] if not has_batch else cur_id[device_id][batch_id]
            # Reduce to scalar
            while hasattr(scenario_id, 'shape') and getattr(scenario_id, 'ndim', 0) > 0:
                scenario_id = scenario_id[0]
            scenario_value = scenario_id.item() if hasattr(scenario_id, 'item') else scenario_id
            scenario_str = str(scenario_value)
            whole_map_by_batch.append(
                loading_data(
                    os.path.join(path_to_map, f'{scenario_str}.npy'),
                    mode='np',
                )
            )
            route_path = os.path.join(path_to_route, f'{scenario_str}.npy')
            if os.path.exists(route_path):
                route_array = loading_data(route_path, mode='np')
            else:
                if scenario_str not in _MISSING_ROUTE_IDS:
                    print(f'[LatentDriver] Missing cached route for {scenario_str}, using zeros.')
                    _MISSING_ROUTE_IDS.add(scenario_str)
                route_array = _ROUTE_FALLBACK_TEMPLATE.copy()
            whole_route_by_batch.append(route_array)
            if intention_label_path is not None:
                label_path = os.path.join(intention_label_path, f'{scenario_str}.txt')
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        intention_label.append(f.readlines()[0])
                else:
                    if scenario_str not in _MISSING_LABEL_IDS:
                        print(f'[LatentDriver] Missing intention label for {scenario_str}, defaulting to empty string.')
                        _MISSING_LABEL_IDS.add(scenario_str)
                    intention_label.append('')
        whole_map_by_device_id.append(np.stack(whole_map_by_batch, axis=0))
        whole_route_by_device_id.append(np.stack(whole_route_by_batch, axis=0))
    road_np = np.stack(whole_map_by_device_id, axis=0)
    route_np = np.stack(whole_route_by_device_id, axis=0)
    return road_np, route_np, intention_label


def get_cache_tl_status_baseline(cur_state: datatypes.SimulatorState,
                                path_to_tl: str):
    cur_id = cur_state._scenario_id.reshape(cur_state.shape)
    tl_by_device = []
    dev_dim = cur_id.shape[0] if hasattr(cur_id, 'shape') else 1
    has_batch = (hasattr(cur_id, 'ndim') and cur_id.ndim >= 2)
    for device_id in range(dev_dim):
        tl_by_batch = []
        batch_range = range(cur_id.shape[1]) if has_batch else range(1)
        for batch_id in batch_range:
            scenario_id = cur_id[device_id] if not has_batch else cur_id[device_id][batch_id]
            while hasattr(scenario_id, 'shape') and getattr(scenario_id, 'ndim', 0) > 0:
                scenario_id = scenario_id[0]
            scenario_value = scenario_id.item() if hasattr(scenario_id, 'item') else scenario_id
            scenario_str = str(scenario_value)
            tl_path = os.path.join(path_to_tl, f'{scenario_str}.npy')
            if os.path.exists(tl_path):
                tl_array = loading_data(tl_path, mode='np')
            else:
                if scenario_str not in _MISSING_ROUTE_IDS:
                    print(f'[LatentDriver] Missing cached tl_status for {scenario_str}, using zeros.')
                    _MISSING_ROUTE_IDS.add(scenario_str)
                tl_array = _TL_FALLBACK_TEMPLATE.copy()
            # Remove padding rows that are all zeros.
            tl_array = np.array(tl_array)
            if tl_array.ndim == 1:
                tl_array = tl_array.reshape(-1, 6)
            valid_mask = ~(np.abs(tl_array).sum(axis=-1) == 0)
            tl_array = tl_array[valid_mask]
            tl_by_batch.append(tl_array)
        # Pad within this device to the max length in the batch for stacking consistency.
        max_len = max((arr.shape[0] for arr in tl_by_batch), default=0)
        padded_batch = []
        for arr in tl_by_batch:
            if arr.shape[0] < max_len:
                pad = np.zeros((max_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            padded_batch.append(arr)
        tl_by_device.append(np.stack(padded_batch, axis=0) if padded_batch else np.zeros((0, 0, 6)))
    return np.stack(tl_by_device, axis=0)


def get_cache_on_route_baseline(cur_state: datatypes.SimulatorState,
                                path_to_map: str):
    cur_id = cur_state._scenario_id.reshape(cur_state.shape)
    masks_by_device = []
    dev_dim = cur_id.shape[0] if hasattr(cur_id, 'shape') else 1
    has_batch = (hasattr(cur_id, 'ndim') and cur_id.ndim >= 2)
    for device_id in range(dev_dim):
        masks_by_batch = []
        batch_range = range(cur_id.shape[1]) if has_batch else range(1)
        for batch_id in batch_range:
            scenario_id = cur_id[device_id] if not has_batch else cur_id[device_id][batch_id]
            while hasattr(scenario_id, 'shape') and getattr(scenario_id, 'ndim', 0) > 0:
                scenario_id = scenario_id[0]
            scenario_value = scenario_id.item() if hasattr(scenario_id, 'item') else scenario_id
            scenario_str = str(scenario_value)
            mask_path = os.path.join(path_to_map, f'{scenario_str}_on_route_mask.npy')
            if os.path.exists(mask_path):
                mask_array = loading_data(mask_path, mode='np')
            else:
                if scenario_str not in _MISSING_ROUTE_IDS:
                    print(f'[LatentDriver] Missing on-route mask for {scenario_str}, using zeros.')
                    _MISSING_ROUTE_IDS.add(scenario_str)
                mask_array = _ON_ROUTE_FALLBACK_TEMPLATE.copy()
            masks_by_batch.append(mask_array)
        masks_by_device.append(np.stack(masks_by_batch, axis=0))
    return np.stack(masks_by_device, axis=0)



def build_discretizer(action_space, seperate=False):
    if seperate:
        discretizer_list = []
        action_ranges = np.array(action_space.action_ranges)
        for i in range(len(action_ranges)):
            discretizer_range = action_ranges[i:i+1]
            discretizer = Discretizer(
                        min_value=discretizer_range[...,0],
                        max_value=discretizer_range[...,1],
                        bins = np.array(action_space.bins[i:i+1],dtype=np.int32),
                        )
            discretizer_list.append(discretizer)
        return discretizer_list
    else:
        discretizer_range = np.array(action_space.action_ranges)
        discretizer = Discretizer(
                    min_value=discretizer_range[...,0],
                    max_value=discretizer_range[...,1],
                    bins = np.array(action_space.bins,dtype=np.int32),
                    )
        return discretizer

def combin_traj(
    simulator_state: datatypes.SimulatorState,
):
    """Infers an action from sim_traj[timestep] to log_traj[timestep + 1].

    Args:
        simulator_state: State of the simulator at the current timestep. Will use
            the `sim_trajectory` and `log_trajectory` fields to calculate an action.
        dynamics_model: Dynamics model whose `inverse` function will be used to
        infer the expert action given the logged states.

    Returns:
        Action that will take the agent from sim_traj[timestep] to
            log_traj[timestep + 1].
    """
    prev_sim_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
        simulator_state.sim_trajectory, simulator_state.timestep, 1, axis=-1
    )
    next_logged_traj = datatypes.dynamic_slice(  # pytype: disable=wrong-arg-types  # jax-ndarray
        simulator_state.log_trajectory, simulator_state.timestep + 1, 1, axis=-1
    )
    combined_traj = jax.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=-1),
        prev_sim_traj,
        next_logged_traj,
    )
    return combined_traj
