import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

# Create racecar instance
rc = racecar_core.create_racecar()

# GaussianMap class definition
class GaussianMap:
    def __init__(self, x_res=300, y_res=300, sigma=10, decay_rate=0.99):
        self.x_res = x_res
        self.y_res = y_res
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.gaussian_map = np.zeros((x_res, y_res))
        self.x_center = x_res // 2
        self.y_center = y_res // 2

    def update_gaussian_map(self, lidar_samples):
        self.gaussian_map *= self.decay_rate
        num_samples = len(lidar_samples)
        angles = np.linspace(0, 2 * np.pi, num_samples)

        for i, distance in enumerate(lidar_samples):
            if distance == 0:
                continue

            angle = angles[i]
            # Switch x and y to correctly align the visualized map
            y = int(self.x_center + distance * np.cos(angle))
            x = int(self.y_center + distance * np.sin(angle))  # Notice the negative sign here to flip the axis

            if 0 <= x < self.x_res and 0 <= y < self.y_res:
                xv, yv = np.meshgrid(np.arange(self.x_res), np.arange(self.y_res))
                gaussian = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * self.sigma ** 2))
                self.gaussian_map += gaussian

    def visualize_gaussian_map(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.gaussian_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title('Gaussian Map with Decay')
        plt.show()

def update_lidar_and_visualize():
    lidar_samples = rc.lidar.get_samples()
    if lidar_samples is not None:
        gaussian_map.update_gaussian_map(lidar_samples)

    # Plot the Gaussian map every second
    current_time = time.time()
    if current_time - update_lidar_and_visualize.last_plot_time >= 1.0:
        gaussian_map.visualize_gaussian_map()
        update_lidar_and_visualize.last_plot_time = current_time

update_lidar_and_visualize.last_plot_time = time.time()

def start():
    print(">> Visualizing Lidar-Based Gaussian Map")

def update():
    update_lidar_and_visualize()

# Initialize Gaussian map
gaussian_map = GaussianMap(sigma=8, decay_rate=0.98)

# Register start and update functions
rc.set_start_update(start, update)
rc.go()
