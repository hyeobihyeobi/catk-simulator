import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

class DebugVisualisation():
    def __init__(self):
        self.save_dir = "vis_results/debug_visualisation/"
        self.save_file = os.path.join(self.save_dir, f"debug_vis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        os.makedirs(self.save_dir, exist_ok=True)

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

    def plot_map(self, map_data, lane_data, batch_idx=0):
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

        lane_xs = []
        lane_ys = []
        for i in range(lane_data.shape[2]):
            x = lane_data[0, 0, i, 1]
            y = lane_data[0, 0, i, 2]
            plt.scatter(x, y, color='black')
            lane_xs.append(x)
            lane_ys.append(y)
#             print(x, '\t', y)

        # 자동으로 모든 점을 커버하도록 축 범위 설정
        all_xs = np.array(map_xs + lane_xs)
        all_ys = np.array(map_ys + lane_ys)
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
