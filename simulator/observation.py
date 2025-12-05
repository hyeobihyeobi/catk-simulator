import jax
import jax.numpy as jnp
from waymax import datatypes
import numpy as np
from waymax.datatypes import observation
import chex

from waymax.datatypes.constant import TIME_INTERVAL

from debug_visualisation import DebugVisualisation

def drop_zero_on_roadobs(roadgraph_obs:np.array,
                            max_roadgraph_segments:int = 10000):
    num_device,B,_,attribute = roadgraph_obs.shape

    vali_mask = np.where(roadgraph_obs.sum(-1)!=0,True,False)
    # (num_device,B,1000) -> (num_device*B,1000)
    vali_mask = vali_mask.reshape(num_device*B,-1)
    roadgraph_obs = roadgraph_obs.reshape(num_device*B,-1,attribute)
    def flip_N(A,N):
        '''
            if segments larger than max, then drop, else pad
        '''
        assert A.ndim == 1
        if N > 0:
            # Find the indices of `False` values in `A`
            false_indices = np.where(A == False)[0]
            # Check if the number of `False` values is greater than or equal to N
            if len(false_indices) >= N:
                # Change the first N `False` values to `True`
                A[false_indices[:N]] = True 
        elif N<0:
            # Find the indices of `True` values in `A`
            true_indices = np.where(A == True)[0]
            # Check if the number of `True` values is greater than or equal to -N
            if len(true_indices) >= -N:
                # Change the first -N `True` values to `False`
                A[true_indices[:-N]] = False
        else:
            pass
    # (bs,)
    valid_nums = vali_mask.sum(-1)
    for bs_id in range(B*num_device):
        # if valid_nums[bs_id] < cfg.max_roadgraph_segments:
        # modified in place
        flip_N(vali_mask[bs_id],max_roadgraph_segments-valid_nums[bs_id])
        # else:
        #     raise ValueError('max_roadgraph_segments should be larger than {}'.format(valid_nums[bs_id]))
    roadgraph_obs = roadgraph_obs[vali_mask].reshape(num_device,B,-1,attribute)
    return roadgraph_obs
def preprocess_data_dist_jnp(data: dict[jnp.array],
                            max_roadgraph_segments: int = 10000):
    '''
        The data dict is updated in place here to numpy
        TODO: write max_roadgraph_segments to config
    '''
    type_route_seg = np.array(data['route_segments'])
    type_vehicles = np.array(data['vehicle_segments'])
    type_roadobs = np.array(data['roadgraph_obs'])
    his_veh_trajs = np.array(data['his_veh_trajs'])
    type_roadobs = drop_zero_on_roadobs(type_roadobs,max_roadgraph_segments)

    point_on_route = np.array(data['point_on_route'])
    point_tl_status = np.array(data['point_tl_status'])
    point_has_speed_limit = np.array(data['point_has_speed_limit'])

#     point_position = np.array(data['point_position'])
#     point_vector = np.array(data['point_vector'])
#     point_side = np.array(data['point_side'])
#     point_orientation = np.array(data['point_orientation'])
#     polygon_center = np.array(data['polygon_center'])
#     polygon_position = np.array(data['polygon_position'])
#     polygon_orientation = np.array(data['polygon_orientation'])
#     polygon_type = np.array(data['polygon_type'])
#     polygon_on_route = np.array(data['polygon_on_route'])
#     polygon_tl_status = np.array(data['polygon_tl_status'])
#     polygon_speed_limit = np.array(data['polygon_speed_limit'])
#     polygon_has_speed_limit = np.array(data['polygon_has_speed_limit'])
#     polygon_road_block_id = np.array(data['polygon_road_block_id'])
    valid_mask = np.array(data['valid_mask'])

    # update the data
    data['roadgraph_obs'] = type_roadobs
    data['route_segments'] = type_route_seg
    data['vehicle_segments'] = type_vehicles
    data['his_veh_trajs'] = his_veh_trajs
    data['point_on_route'] = point_on_route
    data['point_tl_status'] = point_tl_status
    data['point_has_speed_limit'] = point_has_speed_limit
    # TODO: Create polygon data
#     data['point_position'] = point_position
#     data['point_vector'] = point_vector
#     data['point_side'] = point_side
#     data['point_orientation'] = point_orientation
#     data['polygon_center'] = polygon_center
#     data['polygon_position'] = polygon_position
#     data['polygon_orientation'] = polygon_orientation
#     data['polygon_type'] = polygon_type
#     data['polygon_on_route'] = polygon_on_route
#     data['polygon_tl_status'] = polygon_tl_status
#     data['polygon_speed_limit'] = polygon_speed_limit
#     data['polygon_has_speed_limit'] = polygon_has_speed_limit
#     data['polygon_road_block_id'] = polygon_road_block_id
    data['valid_mask'] = valid_mask

#     obs = np.concatenate([type_route_seg, type_vehicles, type_roadobs], axis=2,dtype=np.float32)
#     data['obs'] = obs
    # obs [num_devices, collected bs, numbers of types, 7]
    return data
def get_padding_mask(array):
    return array.sum(axis=-1) == 0
def get_vehicle_obs(sdc_obs, timestep):
    # modified for time-step
    # sdc_obs.trajectory.xy.shape [num_gpus,bs,objs,timesteps,2]
    valid_mask = sdc_obs.trajectory.valid[...,:,:timestep,jnp.newaxis]
    xy = sdc_obs.trajectory.xy[...,:,:timestep,:] * valid_mask
    # speed [objs,1]
#     speed = sdc_obs.trajectory.speed[...,:,:timestep,jnp.newaxis] * valid_mask
    vel_xy = sdc_obs.trajectory.vel_xy[...,:timestep,:] * valid_mask

    acc_xy = jnp.diff(vel_xy, axis=-2, prepend=vel_xy[..., :1, :]) / TIME_INTERVAL
    acc_xy = acc_xy * valid_mask
    # yaw [objs,1]
    yaw = sdc_obs.trajectory.yaw[...,:,:timestep,jnp.newaxis] * 180 / np.pi * valid_mask
    width = sdc_obs.trajectory.width[...,:,:timestep,jnp.newaxis] * valid_mask
    length = sdc_obs.trajectory.length[...,:,:timestep,jnp.newaxis] * valid_mask
    vehicle_obs = jnp.concatenate([xy, width, length, yaw, vel_xy, acc_xy, valid_mask], axis=-1)
    return vehicle_obs
def downsampled_elements_transformation(elements,
                                        pose_global2ego,
                                        sdc_yaw,):
    elements = jnp.array(elements)
    # (bs, max_roadgraph_segments, 6)
    elements_shape = elements.shape
    # (bs, max_roadgraph_segments, 6) -> (1,bs,1,max_roadgraph_segments,6)
    elements = elements[:,jnp.newaxis,...]
    unpad_mask = jnp.where(elements.sum(axis=-1) != 0, True, False)
    # transform xy into ego frame (1,bs,1,max_roadgraph_segments,2)
    transformed_xy = observation.geometry.transform_points(
        pts=elements[...,0:2],
        pose_matrix=pose_global2ego.matrix)
    # (1,bs,1) -> (1,bs,1,max_roadgraph_segments)
    sdc_yaw_ = jnp.repeat(sdc_yaw[...,jnp.newaxis], elements.shape[-2], axis=-1)
    '''debug'''
    # !!! sdc is in radian the data we store is in degree
    sdc_yaw_ = sdc_yaw_ * 180 / jnp.pi
    transformed_yaw = observation.geometry.transform_yaw(-sdc_yaw_,elements[...,-2])
    # (1,bs,1,max_roadgraph_segments) -> (1,bs,1,max_roadgraph_segments,1)
    transformed_yaw = transformed_yaw[...,jnp.newaxis]
    # (1,bs,1,max_roadgraph_segments,6)
    new_elements = jnp.concatenate([transformed_xy,
                                    elements[...,2:4],
                                    transformed_yaw,
                                    elements[...,5:6]],axis=-1)
    return new_elements,unpad_mask

def new_downsampled_elements_transformation(elements,
                                        pose_global2ego,
                                        sdc_yaw,):
    """
    Transform elements with layout (x, y, yaw, type, id) into ego frame.
    """
    elements = jnp.array(elements)
    elements = elements[:, jnp.newaxis, ...]
    unpad_mask = jnp.where(elements.sum(axis=-1) != 0, True, False)

    transformed_xy = observation.geometry.transform_points(
        pts=elements[..., 0:2],
        pose_matrix=pose_global2ego.matrix)

    sdc_yaw_ = jnp.repeat(sdc_yaw[..., jnp.newaxis], elements.shape[-2], axis=-1)
    sdc_yaw_ = sdc_yaw_ * 180 / jnp.pi
    transformed_yaw = observation.geometry.transform_yaw(-sdc_yaw_, elements[..., 2])
    transformed_yaw = transformed_yaw[..., jnp.newaxis]

    new_elements = jnp.concatenate([
        transformed_xy,
        transformed_yaw,
        elements[..., 3:4],  # type
        elements[..., 4:5],  # id
    ], axis=-1)
    return new_elements, unpad_mask
def get_obs_from_routeandmap_saved(
                            state:datatypes.SimulatorState,
                            whole_map:np.array,
                            route:np.array,
                            tl_status:np.array,
                            on_route_mask:np.array,
                            vis_distance:list=[50, 50], #for width and height
                            sample_points = 20):
    '''
        TODO: write vis_distance into config
    '''

    # assert len(state.shape)==1  # Disabled for pmap compatibility
    def padding_exceed(array,dis):
        x_exceed_mask = jnp.abs(array[...,0])>dis[0]//2
        y_exceed_mask = jnp.abs(array[...,1])>dis[1]//2
        exceed_mask = jnp.logical_or(x_exceed_mask,y_exceed_mask)
        # array[exceed_mask] *= 0
        array = jnp.where(exceed_mask[...,jnp.newaxis], 0, array)
        return array,exceed_mask

    def add_type_and_reset_padding(array, type_id):
        padding_mask = get_padding_mask(array)
        type_array = jnp.concatenate([jnp.ones(array.shape[:2])[...,jnp.newaxis] * type_id, array], axis=-1)
        # type_array[padding_mask] *= 0
        type_array = jnp.where(padding_mask[...,jnp.newaxis], 0, type_array)
        return type_array
    # whole_map (bs, max_segs, 6)
    B,P = state.roadgraph_points.shape
    # Select the XY position at the current timestep.
    # Shape: (..., num_agents, 2)
    # obj_xy = state.current_sim_trajectory.xy[..., 0, :]
    obj_xy = jnp.squeeze(state.current_sim_trajectory.xy[..., 0:1, :], axis=-2)
    obj_yaw = state.current_sim_trajectory.yaw[..., 0]
    obj_valid = state.current_sim_trajectory.valid[..., 0]

    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    # (1,bs,2)
    sdc_idx_4d = sdc_idx[..., jnp.newaxis, jnp.newaxis]  # add two dimensions
    mask_float = state.object_metadata.is_sdc[..., jnp.newaxis]
    sdc_xy = jnp.sum(obj_xy * mask_float, axis=-2, keepdims=True)
    sdc_yaw = jnp.sum(obj_yaw * state.object_metadata.is_sdc, axis=-1, keepdims=True)
    sdc_valid = jnp.any(jnp.logical_and(obj_valid, state.object_metadata.is_sdc), axis=-1, keepdims=True)
    # The num_obj is 1 because the it is computing the observation for SDC, and
    # there is only 1 SDC per scene.
    num_obj = 1
    time_step = 11
    global_obs = observation.global_observation_from_state(
        state, time_step, num_obj=num_obj
    )
    is_ego = state.object_metadata.is_sdc[..., jnp.newaxis, :]
    global_obs_filter = global_obs.replace(
        is_ego=is_ego,
    )


    pose2d = observation.ObjectPose2D.from_center_and_yaw(
        xy=sdc_xy, yaw=sdc_yaw, valid=sdc_valid
    )
    chex.assert_equal(pose2d.shape, state.shape + (1,))
    sdc_obs = observation.transform_observation(global_obs_filter, pose2d)
    pose_global2ego = observation.combine_two_object_pose_2d(src_pose=global_obs_filter.pose2d, dst_pose=pose2d)
    # for roadgraph
    whole_map_shape = whole_map.shape
#     new_whole_map, unpad_mask_map = downsampled_elements_transformation(whole_map, pose_global2ego, sdc_yaw)
    new_whole_map, unpad_mask_map = new_downsampled_elements_transformation(whole_map, pose_global2ego, sdc_yaw)

#     DebugVisualisation().plot_map_jax(
#         new_whole_map[..., :2],
#         ids=new_whole_map[..., 4],
#         types=new_whole_map[..., 3:4],
#         batch_idx=0,
#     )

    # Align on_route_mask to (B, P, 1) even if loaded with extra channels.
    on_route_mask = jnp.array(on_route_mask)
    B, P = whole_map.shape[:2]
    on_route_mask = on_route_mask.reshape(B, P, -1)[..., :1]
    # ROI_wh = [-vis_distance, vis_distance]
    # ROI_wh = jnp.array(ROI_wh)
    ROI_wh = jnp.array(vis_distance) * 2
    # x
    mask_x = jnp.logical_and(new_whole_map[...,0] >= -ROI_wh[0]//2,
                            new_whole_map[...,0] <= ROI_wh[0]//2)
    # y
    mask_y = jnp.logical_and(new_whole_map[...,1] >= -ROI_wh[1]//2,
                            new_whole_map[...,1] <= ROI_wh[1]//2)
    mask_roi = jnp.logical_and(mask_x, mask_y)

    whole_map_roi = new_whole_map*mask_roi[...,jnp.newaxis]
    whole_map_roi = whole_map_roi * unpad_mask_map[...,jnp.newaxis]
    on_route_roi = on_route_mask * mask_roi[..., jnp.newaxis] * unpad_mask_map[..., jnp.newaxis]

#     DebugVisualisation().plot_map_jax(
#         whole_map_roi[..., :2],
#         ids=whole_map_roi[..., 4],
#         types=whole_map_roi[..., 3:4],
#         batch_idx=0,
#     )

    roadgraph_obs = whole_map_roi.reshape(whole_map_shape)
    # vali_mask = jnp.where(roadgraph_obs.sum(-1)!=0,True,False)
    # # (bs,)
    # valid_nums = vali_mask.sum(-1)
    # for bs_id in range(B):
    #     # if valid_nums[bs_id] < cfg.max_roadgraph_segments:
    #     # modified in place
    #     flip_N(vali_mask[bs_id],max_roadgraph_segments-valid_nums[bs_id])
    #     # else:
    #     #     raise ValueError('max_roadgraph_segments should be larger than {}'.format(valid_nums[bs_id]))
    # roadgraph_obs = roadgraph_obs[vali_mask].reshape(B,-1,6)
    type_roadobs = add_type_and_reset_padding(roadgraph_obs, 3)
    point_on_route = on_route_roi
    point_on_route = jnp.squeeze(point_on_route, axis=-1)
    if point_on_route.ndim == 3 and point_on_route.shape[1] == 1:
        point_on_route = point_on_route[:, 0, :]


    # for vehicle
    vehicle_sgements = get_vehicle_obs(sdc_obs,time_step)
    cur_vehicle_sgements = vehicle_sgements[...,-1,:]

    veh_segs, vehicle_exceed_masks = padding_exceed(cur_vehicle_sgements,dis=ROI_wh)
    veh_segs = veh_segs.reshape(B,cur_vehicle_sgements.shape[-2],cur_vehicle_sgements.shape[-1])
    # for other agents trajs
    # (bs,num_objs,time_step-1,6)
    his_veh_trajs = vehicle_sgements[...,:-1,:]
    his_types = jnp.ones(his_veh_trajs.shape[:-1])[...,jnp.newaxis] * 2
    his_veh_trajs = jnp.concatenate([his_types, his_veh_trajs], axis=-1)
    # set sdc to false
    vehicle_exceed_masks.at[jnp.linspace(0,state.shape[0]-1,state.shape[0]).astype(int),
                            jnp.linspace(0,B-1,B).astype(int),
                            sdc_idx.reshape(-1)].set(False)
    his_veh_trajs = jnp.where(vehicle_exceed_masks[...,jnp.newaxis,jnp.newaxis],0,his_veh_trajs).reshape((-1,)+his_veh_trajs.shape[2:])

    # type_vehicles [bs,7]
    type_vehicles = add_type_and_reset_padding(veh_segs, 2)
    # set sdc type on type_vehicles into 4
    type_vehicles = type_vehicles.at[jnp.linspace(0,B-1,B).astype(int),sdc_idx.reshape(-1),0].set(4)
    # for route
    route_shape = route.shape
    new_route, unpad_mask = downsampled_elements_transformation(route,pose_global2ego,sdc_yaw)
    new_route = new_route * unpad_mask[...,jnp.newaxis]
    route_obs = new_route.reshape(route_shape)
    type_route_seg = add_type_and_reset_padding(route_obs, 1)

    # Map traffic light status to roadgraph points by matching IDs.
    tl_array = jnp.array(tl_status)
    if tl_array.ndim == 2:
        tl_array = tl_array[jnp.newaxis, ...]
    # valid tl rows: non-zero
    tl_valid = jnp.abs(tl_array).sum(-1) != 0
    tl_ids = tl_array[..., 5]
    tl_states = tl_array[..., 4]
    road_ids = type_roadobs[..., -1]  # (B, P)
    match = (road_ids[..., None] == tl_ids[:, None, :]) & tl_valid[:, None, :]
    mapped_state = jnp.where(match, tl_states[:, None, :], 0.0)
    point_tl_status = mapped_state.max(axis=-1)  # (B, P)
    # for vis sdc_obs
    # sdc_obs = jax.tree_util.tree_map(lambda x: x[0,:,:], sdc_obs)
    point_has_speed_limit = jnp.full((1, 1000), False, dtype=bool)

    # Resample roadgraph polylines to fixed 20 points per polyline (host-side numpy to avoid tracer boolean indexing).
    padding_mask = jnp.array(get_padding_mask(roadgraph_obs))
    valid_mask = ~padding_mask
#     road_obs_arr = jnp.array(jax.device_get(roadgraph_obs))
#     ids_arr = jnp.array(jax.device_get(type_roadobs[..., -1]))
#     sampled_paths = []
#     sampled_types = []
#     sampled_ids = []
#     max_paths = 0

#     for b in range(road_obs_arr.shape[0]):
# #         valid_points = road_obs_arr[b][jnp.where(valid_mask[b])[0]]
# #         valid_ids = ids_arr[b][valid_mask[b]].astype(jnp.int32)
#         valid_points = jnp.where(valid_mask[b][:, None], road_obs_arr[b], 0.0)
#         valid_ids = jnp.where(valid_mask[b], ids_arr[b], -1).astype(jnp.int32)
#         batch_paths = []
#         batch_types = []
#         batch_ids = []
#         for uid in jnp.unique(valid_ids):
#             pts = valid_points[valid_ids == uid]
#             if pts.shape[0] < 2:
#                 continue
#             path_xyz = jnp.concatenate([pts[:, 1:3], jnp.zeros((pts.shape[0], 1), dtype=pts.dtype)], axis=-1)
#             sampled = jnp.array(_interpolate_polyline(jnp.array(path_xyz), 20))
#             batch_paths.append(sampled)
#             batch_types.append(jnp.full((20,), pts[0, -1], dtype=sampled.dtype))
#             batch_ids.append(jnp.full((20,), uid, dtype=jnp.int32))
#         max_paths = max(max_paths, len(batch_paths))
#         sampled_paths.append(batch_paths)
#         sampled_types.append(batch_types)
#         sampled_ids.append(batch_ids)
#     padded_paths = []
#     padded_types = []
#     padded_ids = []
#     for paths, types, ids_list in zip(sampled_paths, sampled_types, sampled_ids):
#         if len(paths) < max_paths:
#             paths = paths + [jnp.zeros((20, 3), dtype=jnp.float32)] * (max_paths - len(paths))
#             types = types + [jnp.zeros((20,), dtype=jnp.float32)] * (max_paths - len(types))
#             ids_list = ids_list + [jnp.zeros((20,), dtype=jnp.int32)] * (max_paths - len(ids_list))
#         padded_paths.append(jnp.stack(paths, axis=0) if paths else jnp.zeros((0, 20, 3), dtype=jnp.float32))
#         padded_types.append(jnp.stack(types, axis=0) if types else jnp.zeros((0, 20), dtype=jnp.float32))
#         padded_ids.append(jnp.stack(ids_list, axis=0) if ids_list else jnp.zeros((0, 20), dtype=jnp.int32))
#     roadgraph_sampled = jnp.array(jnp.stack(padded_paths, axis=0) if padded_paths else jnp.zeros((0, 0, 20, 3), dtype=jnp.float32))
#     roadgraph_sampled_type = jnp.array(jnp.stack(padded_types, axis=0) if padded_types else jnp.zeros((0, 0, 20), dtype=jnp.float32))
#     roadgraph_sampled_id = jnp.array(jnp.stack(padded_ids, axis=0) if padded_ids else jnp.zeros((0, 0, 20), dtype=jnp.int32))



#     roadgraph_sampled = jnp.zeros((roadgraph_obs.shape[0], 0, sample_points, 3), dtype=jnp.float32)
#     roadgraph_sampled_type = jnp.zeros((roadgraph_obs.shape[0], 0, sample_points), dtype=jnp.float32)
#     roadgraph_sampled_id = jnp.zeros((roadgraph_obs.shape[0], 0, sample_points), dtype=jnp.int32)
# 
#     B, M, P = roadgraph_sampled.shape[0], roadgraph_sampled.shape[1], sample_points
#     point_position = jnp.zeros((B, M, 1, P, 2), dtype=jnp.float64)
#     point_vector = jnp.zeros((B, M, 1, P, 2), dtype=jnp.float64)
#     point_side = jnp.zeros((B, M, 1), dtype=jnp.int8)
#     point_orientation = jnp.zeros((B, M, 1, P), dtype=jnp.float64)
#     polygon_center = jnp.zeros((B, M, 3), dtype=jnp.float64)
#     polygon_position = jnp.zeros((B, M, 2), dtype=jnp.float64)
#     polygon_orientation = jnp.zeros((B, M,), dtype=jnp.float64)
#     polygon_type = jnp.zeros((B, M,), dtype=jnp.int8)
#     polygon_on_route = jnp.zeros((B, M,), dtype=bool)
#     polygon_tl_status = jnp.zeros((B, M,), dtype=jnp.int8)
#     polygon_speed_limit = jnp.zeros((B, M,), dtype=jnp.float64)
#     polygon_has_speed_limit = jnp.zeros((B, M,), dtype=bool)
#     polygon_road_block_id = jnp.zeros((B, M,), dtype=jnp.int32)
# 
#     point_position = roadgraph_sampled[..., :2]
#     point_vector = roadgraph_sampled[..., 1: ,:2] - roadgraph_sampled[..., :-1, :2]
#     point_side = jnp.arange(1)
#     point_orientation = roadgraph_sampled[..., 2]
#     polygon_center = roadgraph_sampled[..., sample_points // 2,:]
#     polygon_position = roadgraph_sampled[..., 0, :2]
#     polygon_orientation = roadgraph_sampled[..., 0, 2:3]
#     polygon_type = roadgraph_sampled_type
#     polygon_on_route = point_on_route
#     polygon_tl_status = point_tl_status
#     polygon_speed_limit = point_has_speed_limit
#     polygon_has_speed_limit = point_has_speed_limit
#     polygon_road_block_id = roadgraph_sampled_id
#     valid_mask = jnp.any(roadgraph_sampled.sum(axis=-1) != 0, axis=-1)

    return dict(
        route_segments=type_route_seg,
        vehicle_segments=type_vehicles,
#         roadgraph_obs=type_roadobs,
        roadgraph_obs=roadgraph_obs,
        his_veh_trajs = his_veh_trajs,
        point_on_route=point_on_route,
        point_tl_status=point_tl_status,
        point_has_speed_limit=point_has_speed_limit,
#         roadgraph_sampled=roadgraph_sampled,
#         roadgraph_sampled_type=roadgraph_sampled_type,
#         # traj_obs=traj_obs,
#         # traj_next_stamp=traj_next_stamp,
#         point_position = point_position,
#         point_vector = point_vector,
#         point_side = point_side,
#         point_orientation = point_orientation,
#         polygon_center = polygon_center,
#         polygon_position = polygon_position,
#         polygon_orientation = polygon_orientation,
#         polygon_type = polygon_type,
#         polygon_on_route = polygon_on_route,
#         polygon_tl_status = polygon_tl_status,
#         polygon_speed_limit = polygon_speed_limit,
#         polygon_has_speed_limit = polygon_has_speed_limit,
#         polygon_road_block_id = polygon_road_block_id,
        valid_mask = valid_mask
    ), sdc_obs


def _interpolate_polyline(points: jax.Array, t: int) -> jax.Array:
    """copy from av2-api"""

    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = jnp.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen: jax.Array = jnp.linalg.norm(jnp.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / jnp.sum(chordlen)
    # cumulative arclength

    cumarc: jax.Array = jnp.zeros(len(chordlen) + 1)
#     cumarc[1:] = jnp.cumsum(chordlen)
    cumarc = cumarc.at[1:].set(jnp.cumsum(chordlen))

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: jax.Array = jnp.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
#     tbins[jnp.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
#     tbins[jnp.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1
    idx = jnp.where((tbins <= 0) | (eq_spaced_points <= 0))
    tbins = tbins.at[idx].set(1)
    idx = jnp.where((tbins >= n) | (eq_spaced_points >= 1))
    tbins = tbins.at[idx].set(n - 1)

#     chordlen[tbins - 1] = jnp.where(
#         chordlen[tbins - 1] == 0, chordlen[tbins - 1] + 1e-6, chordlen[tbins - 1]
#     )
    idx = tbins - 1
    chordlen = chordlen.at[idx].set(
        jnp.where(
            (tbins == 0),
            0.0,
            jnp.linalg.norm(eq_spaced_points, axis=-1)
        )
    )

    s = jnp.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: jax.Array = anchors + offsets

    return points_interp

get_obs_from_routeandmap_saved_pmap = jax.pmap(
    get_obs_from_routeandmap_saved,
    static_broadcasted_argnums=(5,),
)
get_obs_from_routeandmap_saved_jit = jax.jit(
    get_obs_from_routeandmap_saved,
    static_argnames=('vis_distance',),
)

# get_obs_from_routeandmap_saved_pmap = get_obs_from_routeandmap_saved
# get_obs_from_routeandmap_saved_jit = get_obs_from_routeandmap_saved
