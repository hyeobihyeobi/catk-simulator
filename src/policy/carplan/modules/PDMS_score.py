from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from tqdm import tqdm
from PIL import Image
import os
import torch.nn.functional as F
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)

import time
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from shapely.ops import unary_union


class PDMS_scorer():
    def __init__(self):
        self.num_poses = 40
        ts = TrajectorySampling(num_poses=self.num_poses, interval_length=0.1)
        self._simulator = PDMSimulator(ts)
        self.debug_activate = False
    
    def get_PDMS_score(self, input_dict):
        # 좌표 변환 변수 설정
        # 미미한 차이만으로도 충돌 여부가 갈리기 때문에 여기선 전부 float64로 쓴다(플로팅 오차 차단)
        av_plans = input_dict["av_plan"].squeeze().double()
        ego_states = input_dict["ego_states"]
        agent_future = input_dict["agent_future"].double()
        agent_mask = input_dict["agent_mask"].double()
        av_shape = input_dict["av_shape"].double()
        agent_shape = input_dict["agent_shape"].double()
        av_future = input_dict["av_future"].double()
        centerline = input_dict['centerline'].double()
        static_ = input_dict["static_"].double()
        static_mask = input_dict["static_mask"].double()
        static_shape = input_dict["static_shape"].double()
        map_pos = input_dict["map_pos"].double()
        map_mask = input_dict["map_mask"].double()
        g_loc = torch.tensor([elem.rear_axle.array for elem in ego_states]).to(av_plans.device)
        g_heading = torch.tensor([elem.rear_axle.heading for elem in ego_states]).unsqueeze(-1).to(av_plans.device)
        g_rot_mat = torch.tensor([[[torch.cos(-elem), -torch.sin(-elem)], \
                                  [torch.sin(-elem), torch.cos(-elem)]] \
                                  for elem in g_heading]).to(av_plans.device).double()

        # av axle 기준 rear 좌표계 상태에서 글로벌 좌표계 변환 및 lqr 연산
        # get_collision 함수의 입력으로 사용하기 위해 rear로 되어 있던 좌표계를 center로 옮겨줌
        av_g_plan_pos = torch.einsum('btc,bcd->btd', av_plans[... ,:2], g_rot_mat) + g_loc.unsqueeze(1)
        av_g_plan_heading = self.do_heading_norm(av_plans[..., 2] + g_heading).unsqueeze(-1)
        av_g_plan = torch.concat((av_g_plan_pos, av_g_plan_heading), dim=-1)
        # 모델의 av 경로 출력을 시뮬레이션에 맞추어 lqr trajectory로 변환해준다
        simulated_av_plan_r_axle = torch.tensor(self._simulator.simulate_proposals_for_batch(av_g_plan, ego_states)).to(av_plans.device).double()
        # center좌표계(collision_map 만드는 용도), rear 좌표계(EP, ), frone 좌표계(DR)@미정임
        rear2center_mag = np.linalg.norm(ego_states[0].center.array - ego_states[0].rear_axle.array) # plus, scalar
        rear2center_vec = torch.concat((torch.cos(simulated_av_plan_r_axle[..., 2]).unsqueeze(-1), 
                                 torch.sin(simulated_av_plan_r_axle[..., 2]).unsqueeze(-1)), dim=-1)*rear2center_mag
        simulated_av_plan_c_axle = simulated_av_plan_r_axle.clone()
        simulated_av_plan_c_axle[..., :2] += rear2center_vec
        
        
        #agent는 axle 기준 이미 center를 따르기 때문에 그대로 글로벌 좌표계로 변환한다
        agent_g_future_pos = torch.einsum('bntc,bcd->bntd', agent_future[..., :2], g_rot_mat) + g_loc.unsqueeze(1).unsqueeze(1)
        agent_g_future_heading = self.do_heading_norm(agent_future[..., 2] +  g_heading.unsqueeze(1))
        agent_g_future = torch.concat((agent_g_future_pos, agent_g_future_heading.unsqueeze(-1)), dim=-1)
        agent_g_future = (agent_g_future * agent_mask.unsqueeze(-1)).double()

        # static은 글로벌 좌표계 변환을 한 후, repeat을 이용해 t축을 증가시켜, agent랑 동일한 모양으로 만든다.
        static_g_pos = torch.einsum('bnc,bcd->bnd', static_, g_rot_mat) + g_loc.unsqueeze(1)
        static_g_heading = (g_heading.unsqueeze(1)).repeat(1, static_g_pos.shape[1], 1)
        static_g = torch.concat((static_g_pos, static_g_heading), dim=-1)
        static_g = (static_g * static_mask.unsqueeze(-1)).double()
        static_g_future = static_g.unsqueeze(2).repeat(1, 1, self.num_poses+1, 1)
        
        # 기타 변수들 글로벌 좌표계로 변환(AV GT)
        av_g_future_pos = torch.einsum('bntc,bcd->bntd', av_future[..., :2], g_rot_mat) + g_loc.unsqueeze(1).unsqueeze(1)
        av_g_future_heading = self.do_heading_norm(av_future[..., 2] + g_heading.unsqueeze(1))
        av_g_future = torch.concat((av_g_future_pos, av_g_future_heading.unsqueeze(-1)), dim=-1)
        
        # 기타 변수들 글로벌 좌표계로 변환(centerline)
        g_centerline = torch.einsum('bntc,bcd->bntd', centerline, g_rot_mat) + g_loc.unsqueeze(1).unsqueeze(1)
        
        # 맵 좌표 글로벌 좌표로 변환
        g_map_pos = torch.einsum('bnktc,bcd->bnktd', map_pos, g_rot_mat) + g_loc.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        # DR 여부 검사
        DR_score = []
        for bs_idx in range(g_map_pos.shape[0]):
            g_map_pos_for_poly = torch.concat((g_map_pos[bs_idx][:, 1, ...],  \
                                                torch.flip(g_map_pos[bs_idx][:, 2, ...], dims=[1]), \
                                                    g_map_pos[bs_idx][:, 1, ...][:, 0, :].unsqueeze(1)), dim=1).detach().cpu().numpy()
            map_mask_for_poly = torch.concat((map_mask[bs_idx], \
                                                torch.flip(map_mask[bs_idx], dims=[1]), \
                                                    map_mask[bs_idx][:, 0].unsqueeze(-1)), dim=1).unsqueeze(-1).detach().cpu().numpy().astype(bool)
            polygons = [Polygon(g_map_pos_for_poly[i][map_mask_for_poly[i].reshape(-1)]) for i in range(g_map_pos_for_poly.shape[0]) if np.sum(map_mask_for_poly[i]) > 4]
            multi_polygon = MultiPolygon(polygons).buffer(0)
            AV_center_traj = simulated_av_plan_c_axle[bs_idx, :, :2].detach().cpu().numpy()
            DR_score.append(multi_polygon.contains(MultiPoint(AV_center_traj)))
        DR_score = (~torch.tensor(DR_score)).to(g_map_pos.device).float().unsqueeze(1)
        
        # SAT 알고리즘을 통해 Ego 및 DR 충돌 벡터화 연산
        collision_result = self.get_collision_map(simulated_av_plan_c_axle[...,:3], simulated_av_plan_r_axle[...,:3], av_shape, agent_g_future, agent_shape, ego_states) # (16, N, T)
        debug_setting = self.debug_activate
        self.debug_activate = False
        collision_result_static = self.get_collision_map(simulated_av_plan_c_axle[...,:3], simulated_av_plan_r_axle[...,:3],av_shape, static_g_future, static_shape, ego_states)
        self.debug_activate = debug_setting        
        
        # Ego_Progress 연산
        EP_score = self.get_EP(g_centerline, av_g_future, simulated_av_plan_r_axle.unsqueeze(1)) # (batch, 1)

        # Comfort 연산
        C_score = self.get_C(simulated_av_plan_r_axle.unsqueeze(1)) # (batch, 1)
        C_score = C_score.double()
        # TTC 연산 # 너무 오래 걸릴 거 같아...
        # 0.2초 단위로 20번 계산
        # TTC_score = self.get_TTC(simulated_av_plan_c_axle[...,:5], av_shape, agent_g_future, agent_shape, ego_states, 0.2) # (batch, 1)
        # TTC_score = 1- TTC_score.double()
        TTC_score=C_score.new_ones(C_score.shape)
        
        # MUL_SCORE 중 하나인 Collision score
        PDMS_COL = self.get_NC(simulated_av_plan_c_axle[...,3:5], collision_result, map_target = "agent") # (batch, 1)
        PDMS_COL_static = self.get_NC(simulated_av_plan_c_axle[...,3:5], collision_result_static, map_target="static") # (batch, 1) # 카운트 아직 구현 못함
        
        
        #DAC score... 아직 구현 못함 근데 이거 안하면 반쪽짜리임
        # PDMS = PDMS_COL * 
        PDMS = (PDMS_COL * PDMS_COL_static * DR_score) * (((5*EP_score+5*TTC_score+2*C_score)+1e-3)/12)
        PDMS = torch.clamp(PDMS, min=0.0, max=1.0)
        
        # print()
        # # 
        # # DR 정보 글로벌 캐쉬로 저장
        # # 
        # ##

        # # 계산된 collision_matrix를 점수화
        
        # PDMS = 0
        # PDMS_COL = 0
        # PDMS_DAC = 0
        # PDMS_EP = 0
        # PDMS_TTC = 0
        # PDMS_C = 0
        return PDMS, (PDMS_COL * PDMS_COL_static), DR_score, EP_score, TTC_score, C_score 

    def do_heading_norm(self, heading):
        norm_head = torch.atan2(torch.sin(heading), torch.cos(heading))
        return norm_head
    def get_collision_map(self, av_trajectory, av_r_trajectory, av_size, agent_trajectory, agent_size, ego_states):
        front_agent_mask = self.compute_front_agent_mask(av_trajectory, av_r_trajectory, agent_trajectory)
        av_boxes = self.compute_rotated_box(av_trajectory.unsqueeze(1), av_size)
        agent_boxes = self.compute_rotated_box(agent_trajectory, agent_size)
        result_map = self.check_collision_sat(av_boxes.clone(), agent_boxes.clone())
        if self.debug_activate:
            num_agent = 20
            start_time = 0
            end_time = 30
            os.makedirs("./exp/pdms_result", exist_ok=1)
            result_map_ = self.check_collision_sat(av_boxes[:, :, start_time:end_time, ...].clone(), agent_boxes[:, :num_agent, start_time:end_time, ...].clone())
            result_map_ = result_map_[0]
            self.visualize_collision_mask(result_map_.detach().cpu().numpy())
            for t_index in tqdm(range(end_time)):
                self.get_box_view(av_boxes[0:1, 0:1, t_index:t_index+1, ...], agent_boxes[0:1, :num_agent, t_index:t_index+1, ...], ego_states[0], t_index)            

        return result_map
 
    def compute_front_agent_mask(self, av_c, av_r, agent_c):
        
        
        return 0
 
    def compute_rotated_box(self, positions, sizes):
        """Compute rotated box corners from positions and sizes."""
        # Extract position and heading
        
        heading = positions[..., 2]
        width, height = sizes[..., 1], sizes[..., 0]

        half_w, half_h = width / 2, height / 2

        # Box corners relative to center (before rotation)
        corners = torch.stack([
            torch.stack([-half_w, -half_h], dim=-1),
            torch.stack([ half_w, -half_h], dim=-1),
            torch.stack([ half_w,  half_h], dim=-1),
            torch.stack([-half_w,  half_h], dim=-1)
        ], dim=-2) # Shape: (batch, 1, 4, 2)
        corners = corners.unsqueeze(2).repeat(1, 1, positions.shape[2], 1, 1) # Shape: (batch, 1, timestep, 4, 2)
        
        # Ensure broadcasting is correct
        # corners = corners.expand(-1, positions.shape[1], -1, -1, -1)  # Align dimensions

        # Rotation matrix
        cos_theta = torch.cos(heading).unsqueeze(-1)
        sin_theta = torch.sin(heading).unsqueeze(-1)
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, sin_theta], dim=-1),
            torch.stack([-sin_theta,  cos_theta], dim=-1)
        ], dim=-2)  # Shape: (batch, 1, timestep, 2, 2)

        # Rotate corners
        rotated_corners = torch.einsum('bntij,bntijq->bntiq',corners, rotation_matrix)

        # Translate corners to center (x, y)
        box_corners = rotated_corners + positions[..., :2].unsqueeze(-2)

        return box_corners

    def check_collision_sat(self, vehicle_boxes, agent_boxes):
        """Check collisions between vehicle and agent boxes using SAT."""
        # Edges of vehicle and agent boxes
        vehicle_edges = torch.cat(
            [vehicle_boxes[..., 1:, :] - vehicle_boxes[..., :-1, :],  # (batch, 1, timestep, 3, 2)
            vehicle_boxes[..., :1, :] - vehicle_boxes[..., -1:, :]],  # Closing edge
            dim=-2)  # Shape: (batch, 1, timestep, 4, 2)

        agent_edges = torch.cat(
            [agent_boxes[..., 1:, :] - agent_boxes[..., :-1, :],  # (batch, N, timestep, 3, 2)
            agent_boxes[..., :1, :] - agent_boxes[..., -1:, :]],  # Closing edge
        dim=-2)  # Shape: (batch, N, timestep, 4, 2)

        # Axes (normal vectors)
        vehicle_axes = torch.cat([-vehicle_edges[..., 1:2], vehicle_edges[..., :1]], dim=-1)  # Shape: (batch, 1, timestep, 4, 2)
        agent_axes = torch.cat([-agent_edges[..., 1:2], agent_edges[..., :1]], dim=-1)  # Shape: (batch, N, timestep, 4, 2)
        axes = torch.cat([vehicle_axes, agent_axes], dim=1)  # Shape: (batch, N + 1, timestep, 4, 2)

        # Normalize axes
        axes = axes / torch.norm(axes, dim=-1, keepdim=True)  # Shape: (batch, N + 1, timestep, 4, 2)

        # Project boxes onto axes
        axes = axes.unsqueeze(1)  # Expand axes to (batch, 1, N + 1, timestep, 4, 2)
        projections = torch.einsum('batck,bantik->bantic', vehicle_boxes, axes)#batch, av(1), timestep, corner, k,    # Shape: (batch, 1, N + 1, timestep, 4,4)
        agent_projections = torch.einsum('bdtck,bantik->bdntic', agent_boxes, axes)  # Shape: (batch, N, N + 1, timestep, 4, 4)

        vehicle_min, _ = projections.min(dim=-1)  # Shape: (batch, 1, N + 1, timestep, 4)
        vehicle_max, _ = projections.max(dim=-1)  # Shape: (batch, 1, N + 1, timestep, 4)

        agent_min, _ = agent_projections.min(dim=-1)  # Shape: (batch, N, N + 1, timestep, 4)
        agent_max, _ = agent_projections.max(dim=-1)  # Shape: (batch, N, N + 1, timestep, 4)

        # Check overlap
        overlap = (vehicle_max >= agent_min) & (agent_max >= vehicle_min)  # Shape: (batch, N, N + 1, timestep, 4)
        collision = overlap.all(dim=2).all(dim=-1)# True if no separating axis, Shape: (batch, N, timestep)

        return collision 

    def get_box_view(self, av_boxes, agent_boxes, ego_state, t_index):
        # 박스 데이터 정의
        # 박스 좌표 추출
        av_box_coords = av_boxes[0, 0, 0].squeeze().cpu().numpy()
        agent_box_coords_list = agent_boxes[0, :, 0].cpu().numpy()

        markers = ['o', 'x', '*', '^']  # AV 박스 꼭지점 모양

        # 시각화
        plt.figure(figsize=(5, 5))
        plt.xlim([ego_state.center.x-25, ego_state.center.x+25])
        plt.ylim([ego_state.center.y-25, ego_state.center.y+25])
        # AV 박스 그리기
        for i, coord in enumerate(av_box_coords):
            plt.scatter(coord[0], coord[1], marker=markers[i], color="red", s = 40)
        plt.plot(*zip(*(av_box_coords.tolist() + [av_box_coords[0]])), label='AV Box', color="red")
        # Agent 박스 그리기
        for index, agent_box_coords in enumerate(agent_box_coords_list):
            plt.plot(*zip(*(agent_box_coords.tolist() + [agent_box_coords[0]])), label=f'{index}')

        # 그래프 꾸미기
        plt.title('AV and Agent Boxes')
        plt.xlabel('X Coordinates')
        plt.ylabel('Y Coordinates')
        plt.legend()
        plt.grid()

        # 이미지 저장
        plot_file = f'./exp/pdms_result/boxes_plot_{t_index}.png'
        plt.savefig(plot_file)
        plt.clf()
        plt.close('all')

        # collision_box.png 읽기
        collision_image_path = './exp/pdms_result/collision_box.png'
        try:
            collision_image = Image.open(collision_image_path)
        except FileNotFoundError:
            print(f"Error: {collision_image_path} not found.")
            return

        # 생성된 박스 시각화 이미지 읽기
        box_image = Image.open(plot_file)

        # 두 이미지를 가로로 병합
        total_width = collision_image.width + box_image.width
        max_height = max(collision_image.height, box_image.height)

        merged_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        merged_image.paste(box_image, (0, 0))
        merged_image.paste(collision_image, (box_image.width, 0))

        # 병합된 이미지 저장
        merged_output_path = f'./exp/pdms_result/merged_boxes_plot_{t_index}.png'
        merged_image.save(merged_output_path)
        # print(f"Saved merged image to {merged_output_path}")


    def visualize_collision_mask(self, mask):
        """
        Visualize a collision mask as a heatmap using matplotlib.
        
        Parameters:
        - mask: (1, num_agent, timestep) Collision mask tensor or array.
        """
        # Squeeze the first dimension if needed
        # mask = mask.squeeze(0)  # Shape: (num_agent, timestep)
        
        plt.figure(figsize=(8, 5))  # Set the figure size
        plt.imshow(mask, cmap='coolwarm', interpolation='nearest', aspect='auto')
        
        # Add ticks and labels
        plt.xticks(range(mask.shape[1]), [f"{t}" for t in range(mask.shape[1])])  # Label timesteps
        plt.yticks(range(mask.shape[0]), [f"A{a}" for a in range(mask.shape[0])])  # Label agents

        # Add gridlines
        plt.gca().set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
        plt.grid(which="minor", color="black", linestyle='-', linewidth=0.5)

        plt.savefig(f'./exp/pdms_result/collision_box.png')
        plt.clf()
        plt.close('all')

    # Drive area를 벗어나나요?
    # Non DR은 복잡한 모양을 가지기 때문에 SAT 알고리즘으로는 판단 불가
    # 아래 3개 방안이 있을 거 같은데..
    # Non DR을 단순한 모양 여러개로 쪼개기
    # AV의 center가 모든 LANE(단순한 모양)에 걸리지 않는 순간이 있는지 여부를 판별 <- 도로를 일종의 agent로 취급하고, agent와 다르게 충돌이 없는 step이 존재하는지 확인하는 방식
    # 모든 도로 요소와 agent간의 거리를 측정 후 가장 가까운 거리가 일정 Threshold를 벗어나면 Non dr로 판별
    def get_DAC(self,):
        pass
    
    # No Collision 
    # 빡세게 검사는 어렵지만,, AV가 움직이는 중에 충돌은 모두 유효 collision으로 분류
    # 추후 AV와 Agent간 위상 관계를 파악해서, AV 뒤쪽에 있는 Agent의 충돌은 무시하도록 업그레이드 진행
    def get_NC(self,plan_speed, collision_map, map_target):
        scalar_speed = torch.norm(plan_speed, dim=-1)
        scalar_speed = scalar_speed.unsqueeze(1).repeat(1,collision_map.shape[1],1)
        # AV가 움직이지 않는 상태에서의 충돌은 무시
        # 기준은 0.5m/s를 기준으로 # 0.5m/s는 근거 없는 수치라 더 좋은 reference 찾을 필요 있음
        speed_mask = scalar_speed>0.5
        result = collision_map * speed_mask
        result_ = torch.any(result, axis=-1) # time 축 감소
        num_col_object = torch.sum(result_, axis=-1)
        
        result__ = torch.any(result_, axis=1).unsqueeze(1)
        
        if map_target == "static":
            result___ = 1 - (result__.double() * (0.5 * num_col_object.unsqueeze(1)))
            result___ = torch.clamp(result___, min=0.0, max=1.0)
            return result___
        else:    
            return 1 - (result__.double())
    
    # Ego_progress
    # 최대 20cm 오차
    def get_EP(self,centerline_, av_future, av_plan):
        # 마지막 차원(2)을 분리: (batch, channels, 120)로 변환
        centerline_x = centerline_[..., 0] # x 좌표
        centerline_y = centerline_[..., 1]  # y 좌표
        # 보간 수행 (120 -> 1200)
        centerline_x_resized = F.interpolate(centerline_x, size=1200, mode='linear', align_corners=True).unsqueeze(-1)
        centerline_y_resized = F.interpolate(centerline_y, size=1200, mode='linear', align_corners=True).unsqueeze(-1)
        centerline = torch.cat([centerline_x_resized, centerline_y_resized], dim=-1)
        
        GT_S_p = av_future[:,:,0, :2].unsqueeze(2)
        GT_F_p = av_future[:,:,-1, :2].unsqueeze(2)
        plan_F_p = av_plan[:,:,-1, :2].unsqueeze(2)
        GT_distances = torch.cdist(GT_F_p, centerline)
        GT_closest_indices = torch.argmin(GT_distances, dim=-1).squeeze(-1)
        GT_s_distances = torch.cdist(GT_S_p, centerline)
        GT_s_closest_indices = torch.argmin(GT_s_distances, dim=-1).squeeze(-1)
        
        plan_distances = torch.cdist(plan_F_p, centerline)
        plan_closest_indices = torch.argmin(plan_distances, dim=-1).squeeze(-1)
        
        GT_progress = GT_closest_indices - GT_s_closest_indices
        plan_progress = plan_closest_indices - GT_s_closest_indices
        progress_mask = GT_progress < 50 # 5.0m NAVSIM 정의
        
        progress_score = (plan_progress+1e-2) / (GT_progress+1e-2)
        progress_score_ = torch.clamp(progress_score, max=1.0) * (~progress_mask) + progress_mask
        
        return progress_score_
    
    # Time to collision # 직진으로 판단
    def get_TTC(self, av_trajectory, av_size, agent_trajectory, agent_size, ego_states, s):
        num_speed_mul = s*10
        num_iter = int(4/s)
        speed = av_trajectory[:, 1:, :2] -av_trajectory[:, :-1, :2]*num_speed_mul
        av_trajectory = av_trajectory[..., :3]
        result = []
        for i in range(num_iter):
            av_trajectory[:, :-1, :2] += speed
            av_boxes = self.compute_rotated_box(av_trajectory.unsqueeze(1), av_size)
            agent_boxes = self.compute_rotated_box(agent_trajectory, agent_size)
            result_map = self.check_collision_sat(av_boxes.clone(), agent_boxes.clone())
            result.append(result_map)
        result_ = torch.stack(result)
        result__ = torch.any(result_, dim=0).to(av_trajectory.device)
        result___ = torch.any(result__, dim=-1)
        result____ = torch.any(result___, dim=-1).unsqueeze(-1)
        
        return result____

    # Comfort
    def get_C(self,av_plans):
        time_point_s = (np.arange(0,41).astype(np.float64)* 0.1)
        av_plans_np = av_plans.detach().cpu().numpy()
        result = []
        for av_plan in av_plans_np:
            is_comfortable = ego_is_comfortable(av_plan, time_point_s)
            result.append(is_comfortable)
        result_ = np.all(np.array(result), axis=-1)
        
        result__ = torch.tensor(result_).to(av_plans.device)
        
        return result__
    
    
        


    # 생각보다 할 수 있는 일이 많을 거 같음
    # loss는 직접적으로 흘릴수 없지만, 시뮬레이션상에서 AV의 움직임까지 고려한 상황에서 경로 생성에 중요한 요소를 판별하는 방법 중 하나가 될 수 있음
    # 학습 중에 하는 것이 오래 걸린다면, loss로 어려운 시나리오를 판별, 어려운 시나리오에 대해서만 적용해보는 것도...
    # 1. 충돌하는 객체는 AV의 Trajectory에 영향을 많이 주는 요소임 즉 AV의 입장에서 Inform한 객체라고 할 수 있음 해당 객체의 정보를 MOE의 라우팅에 사용
    # 2. AV의 미래 경로를 multi modal로 한다면, 주변 객체마다 충돌이 발생하는 modal을 할당 가능 , 이때 충돌이 발생하는 modal이 많을수록 해당 객체는 AV의 입장에서 충돌할 가능성이 높은 객체라고 할 수 있음
    # 2-1. 충돌할 가능성이 높은 객체를 다시 키 벨류로 주고, 경로를 학습 단계에서 refinement 가능 -> Gameformer랑 다르게 중요한 요소만 refine에 활용한다는점
    # 3. LQR로 만들어진 경로에서의 충돌 판단, 모델이 만든 경로에서의 충돌 판단 / LQR로 만들어진 경로에서만 충돌이 일어나는 경우에 대해서만 LOSS를 흘려주는 보조 모델 설계
    # 4. 충돌하는 시나리오는 취약 시나리오 학습 하면서 충돌이 일어나는 시나리오를 반복적으로 노출시키는 전략