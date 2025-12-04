import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import os
import sys

class DebugVisualisation():
    def __init__(self):
        self.save_dir = "vis_results/debug_visualisation/"
        self.save_file = os.path.join(self.save_dir, f"debug_vis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        os.makedirs(self.save_dir, exist_ok=True)
#         self.colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.colours = []

        self.colours += ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        self.colours += list(matplotlib.colors.BASE_COLORS.keys())
        self.colours += list(matplotlib.colors.TABLEAU_COLORS.keys())
        self.colours += list(matplotlib.colors.CSS4_COLORS.keys())

        self.colours = list(dict.fromkeys(self.colours))

    def plot_episode(self, states, predictions, batch_idx=0):
        """
        states: (T, N, 7) - ego vehicle states over time
        predictions: list of (N, action_dim) - predicted actions at each timestep
        batch_idx: index of the batch to visualize
        """
        T = states.shape[0]
        plt.figure(figsize=(10, 10))
        for t in range(T):
            state = states[t, batch_idx]
            plt.plot(state[0], state[1], 'bo')  # Plot ego position
            if t < len(predictions):
                pred = predictions[t][batch_idx]
                plt.arrow(state[0], state[1], pred[0], pred[1], head_width=0.5, head_length=1.0, fc='r', ec='r')
        plt.title(f'Episode Visualization for Batch Index {batch_idx}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid()
        plt.savefig(self.save_file)

    def plot_map(self, map_data, lane_data=None, batch_idx=0):
        """
        map_data: map information to plot
        predictions: list of (N, action_dim) - predicted actions at each timestep
        batch_idx: index of the batch to visualize
        """
        plt.figure(figsize=(10, 10))
        # Plot map data (roads, lanes, etc.)
        map_xs = []
        map_ys = []
        for i in range(map_data.shape[2]):
            x = map_data[0, 0, i, 1]
            y = map_data[0, 0, i, 2]
            plt.scatter(x, y, color='black')
            map_xs.append(x)
            map_ys.append(y)
#             print(x, '\t', y)

        if lane_data is not None:
            lane_xs = []
            lane_ys = []
            for i in range(lane_data.shape[2]):
                x = lane_data[0, 0, i, 1]
                y = lane_data[0, 0, i, 2]
                plt.scatter(x, y, color='black')
                lane_xs.append(x)
                lane_ys.append(y)
#                 print(x, '\t', y)

            # 자동으로 모든 점을 커버하도록 축 범위 설정
            all_xs = np.array(map_xs + lane_xs)
            all_ys = np.array(map_ys + lane_ys)
        else:
            all_xs = np.array(map_xs)
            all_ys = np.array(map_ys)
        # NaN 등 제외
        valid_mask = ~np.isnan(all_xs) & ~np.isnan(all_ys)
        if np.any(valid_mask):
            vx = all_xs[valid_mask]
            vy = all_ys[valid_mask]
            x_min, x_max = float(vx.min()), float(vx.max())
            y_min, y_max = float(vy.min()), float(vy.max())
            # 약간의 여백 추가
            x_pad = max((x_max - x_min) * 0.05, 1.0)
            y_pad = max((y_max - y_min) * 0.05, 1.0)
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(y_min - y_pad, y_max + y_pad)

        plt.title(f'Map Visualization for Batch Index {batch_idx}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid()
        plt.savefig(self.save_file)


    def plot_map_jax(self, map_data, lane_data=None, ids=None, types=None, batch_idx=0):
        """
        map_data: map information to plot
        predictions: list of (N, action_dim) - predicted actions at each timestep
        batch_idx: index of the batch to visualize
        """
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # Plot map data (roads, lanes, etc.)
        map_xs = []
        map_ys = []
        map_data = np.array(map_data)
        is_first = True
        colour_change_idx = 0
        id = 0
        for i in range(map_data.shape[2]):
#             if types is None or (int(types[0, i, 0]) != 1 and int(types[0, i, 0]) != 2 and int(types[0, i, 0]) != 18):
#             if types is None or (int(types[0, i, 0]) != 2 and int(types[0, i, 0]) != 15):
#                 # Skip lane points for road plotting
#                 continue
#             print(types[0, i, 0], end=' ')
            x = map_data[0, 0, i, 0]
            y = map_data[0, 0, i, 1]
            if np.abs(x) > 5000 or np.abs(y) > 5000:
                continue
            if x == -1 or y == -1:
                if is_first:
                    is_first = False
                    print('valid points : ', i)
                    sys.stdout.flush()
                break
            plt.scatter(x, y, color=self.colours[colour_change_idx % len(self.colours)], s=0.1)
            map_xs.append(x)
            map_ys.append(y)
            if ids is not None:
                current_id = int(ids[0, 0, i])
                if current_id == 0:
                    continue
#                 print(current_id, end=' ')
                if current_id != id:
                    id = current_id
                    colour_change_idx += 1
#                     print("ID has been changed to:", id)
#                     print("ID index has been changed to:", self.colours[colour_change_idx % len(self.colours)])
#                     print()
#                     sys.stdout.flush()

        is_first = True
        if lane_data is not None:
            lane_xs = []
            lane_ys = []
            lane_data = np.array(lane_data)
            for i in range(lane_data.shape[2]):
                if x == -1 or y == -1:
                    if is_first:
                        is_first = False
                        print(i)
                        sys.stdout.flush()
                    break
                x = lane_data[0, 0, i, 1]
                y = lane_data[0, 0, i, 2]
                plt.scatter(x, y, color='black', s=0.1)
                lane_xs.append(x)
                lane_ys.append(y)
#                 print(x, '\t', y)

            # 자동으로 모든 점을 커버하도록 축 범위 설정
            all_xs = np.array(map_xs + lane_xs)
            all_ys = np.array(map_ys + lane_ys)
        else:
            all_xs = np.array(map_xs)
            all_ys = np.array(map_ys)
        # NaN 등 제외
        valid_mask = ~np.isnan(all_xs) & ~np.isnan(all_ys)
        if np.any(valid_mask):
            vx = all_xs[valid_mask]
            vy = all_ys[valid_mask]
            x_min, x_max = float(vx.min()), float(vx.max())
            y_min, y_max = float(vy.min()), float(vy.max())
            # 약간의 여백 추가
            x_pad = max((x_max - x_min) * 0.05, 1.0)
            y_pad = max((y_max - y_min) * 0.05, 1.0)
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(y_min - y_pad, y_max + y_pad)

        plt.title(f'Map Visualization for Batch Index {batch_idx}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid()
        plt.savefig(self.save_file)

    def plot_map_jax_by_step(self, map_data, lane_data=None, ids=None, types=None, batch_idx=0):
        """
        map_data: map information to plot
        predictions: list of (N, action_dim) - predicted actions at each timestep
        batch_idx: index of the batch to visualize
        """
        plt.figure(figsize=(10, 10))
        # Plot map data (roads, lanes, etc.)
        map_xs = []
        map_ys = []
        map_data = np.array(map_data)
        is_first = True
        colour_change_idx = 0
        previous_id = None
        base_name, ext = os.path.splitext(self.save_file)
        intermediate_idx = 0

        def save_intermediate_progress(tag):
            nonlocal intermediate_idx
            if not map_xs:
                return
            xs = np.array(map_xs)
            ys = np.array(map_ys)
            valid_mask = ~np.isnan(xs) & ~np.isnan(ys)
            if np.any(valid_mask):
                vx = xs[valid_mask]
                vy = ys[valid_mask]
                x_min, x_max = float(vx.min()), float(vx.max())
                y_min, y_max = float(vy.min()), float(vy.max())
                x_pad = max((x_max - x_min) * 0.05, 1.0)
                y_pad = max((y_max - y_min) * 0.05, 1.0)
                plt.xlim(x_min - x_pad, x_max + x_pad)
                plt.ylim(y_min - y_pad, y_max + y_pad)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.grid()
            plt.title(f'Map Visualization for Batch Index {batch_idx} ({tag})')
            snapshot_path = f"{base_name}_step_{intermediate_idx:03d}_{tag}{ext}"
            plt.savefig(snapshot_path)
            intermediate_idx += 1
        for i in range(map_data.shape[2]):
            x = map_data[0, 0, i, 0]
            y = map_data[0, 0, i, 1]
            if x == -1 or y == -1:
                if is_first:
                    is_first = False
                    print('valid points : ', i)
                    sys.stdout.flush()
                break
            snapshot_tag = None
            if ids is not None:
                current_id = ids[0, i, 0]
#                 print(current_id, end=' ')
                if previous_id is None:
                    previous_id = current_id
                elif current_id != previous_id:
                    snapshot_tag = f"id_{previous_id}_to_{current_id}_idx_{i}"
                    previous_id = current_id
                    colour_change_idx += 1
            plt.scatter(x, y, color=self.colours[colour_change_idx % len(self.colours)], s=0.1)
            map_xs.append(x)
            map_ys.append(y)
            if snapshot_tag is not None:
                save_intermediate_progress(snapshot_tag)
#                     print("ID has been changed to:", current_id)
#                     print("ID index has been changed to:", self.colours[colour_change_idx % len(self.colours)])
#                     print()
#                     sys.stdout.flush()

        is_first = True
        if lane_data is not None:
            lane_xs = []
            lane_ys = []
            lane_data = np.array(lane_data)
            for i in range(lane_data.shape[2]):
                if x == -1 or y == -1:
                    if is_first:
                        is_first = False
                        print(i)
                        sys.stdout.flush()
                    break
                x = lane_data[0, 0, i, 1]
                y = lane_data[0, 0, i, 2]
                plt.scatter(x, y, color='black', s=0.1)
                lane_xs.append(x)
                lane_ys.append(y)
#                 print(x, '\t', y)

            # 자동으로 모든 점을 커버하도록 축 범위 설정
            all_xs = np.array(map_xs + lane_xs)
            all_ys = np.array(map_ys + lane_ys)
        else:
            all_xs = np.array(map_xs)
            all_ys = np.array(map_ys)
        # NaN 등 제외
        valid_mask = ~np.isnan(all_xs) & ~np.isnan(all_ys)
        if np.any(valid_mask):
            vx = all_xs[valid_mask]
            vy = all_ys[valid_mask]
            x_min, x_max = float(vx.min()), float(vx.max())
            y_min, y_max = float(vy.min()), float(vy.max())
            # 약간의 여백 추가
            x_pad = max((x_max - x_min) * 0.05, 1.0)
            y_pad = max((y_max - y_min) * 0.05, 1.0)
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(y_min - y_pad, y_max + y_pad)

        plt.title(f'Map Visualization for Batch Index {batch_idx}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid()
        plt.savefig(self.save_file)
