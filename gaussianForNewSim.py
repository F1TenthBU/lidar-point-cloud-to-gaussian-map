# Example usage of RacecarMLAgent class:
import time
from racecar_ml_agent import RacecarMLAgent
import os
import numpy as np
import matplotlib.pyplot as plt

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Parent directory path:", parent_directory)

env_path = parent_directory + "/Mac.app"
racecar = RacecarMLAgent(env_path, time_scale=1.0)

speed = 0
angle = 0


########################################################################################
# GaussianMap Class
########################################################################################
class GaussianMap:
    def __init__(self, x_res=300, y_res=300, sigma=6, decay_rate=0.995):
        self.x_res = x_res
        self.y_res = y_res
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.gaussian_map = np.zeros((x_res, y_res))
        self.x_center = x_res // 2
        self.y_center = y_res // 2

    def apply_gaussian(self, distance, angle):
        if distance == 0:
            return

        # Correct x and y calculations
        y = int(self.y_center - distance * np.cos(angle))  # Adjust y-axis
        x = int(self.x_center + distance * np.sin(angle))  # Adjust x-axis

        if 0 <= x < self.x_res and 0 <= y < self.y_res:
            xv, yv = np.meshgrid(np.arange(self.x_res), np.arange(self.y_res))
            gaussian = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * self.sigma ** 2))
            self.gaussian_map += gaussian

    def update_gaussian_map(self, lidar_samples):
        self.gaussian_map = np.zeros((self.x_res, self.y_res))

        num_samples = len(lidar_samples)
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)

        # Log points near left (-90 degrees) and right (90 degrees)
        left_bound_idx = np.argmin(np.abs(angles - (np.pi / 2)))
        right_bound_idx = np.argmin(np.abs(angles - (3 * np.pi / 2)))
        front_bound_idx = np.argmin(np.abs(angles - 0))  # 0° for front
        back_bound_idx = np.argmin(np.abs(angles - np.pi))  # 180° for back 

        print(f"Front bound LiDAR distance: {lidar_samples[front_bound_idx]} at angle {np.degrees(angles[front_bound_idx]):.2f}")
        print(f"Back bound LiDAR distance: {lidar_samples[back_bound_idx]} at angle {np.degrees(angles[back_bound_idx]):.2f}")
        print(f"Left bound LiDAR distance: {lidar_samples[left_bound_idx]} at angle {np.degrees(angles[left_bound_idx]):.2f}")
        print(f"Right bound LiDAR distance: {lidar_samples[right_bound_idx]} at angle {np.degrees(angles[right_bound_idx]):.2f}")

        radius = int(3 * self.sigma)
        xv, yv = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        gaussian_template = np.exp(-(xv ** 2 + yv ** 2) / (2 * self.sigma ** 2))

        for i, distance in enumerate(lidar_samples):
            if distance == 0:
                continue

            angle = angles[i]
            y = int(self.y_center - distance * np.cos(angle))  # Adjust y-axis
            x = int(self.x_center + distance * np.sin(angle))  # Adjust x-axis

            if 0 <= x < self.x_res and 0 <= y < self.y_res:
                x_start = max(0, x - radius)
                x_end = min(self.x_res, x + radius + 1)
                y_start = max(0, y - radius)
                y_end = min(self.y_res, y + radius + 1)

                template_x_start = max(0, radius - x)
                template_y_start = max(0, radius - y)
                template_x_end = template_x_start + (x_end - x_start)
                template_y_end = template_y_start + (y_end - y_start)

                self.gaussian_map[y_start:y_end, x_start:x_end] += gaussian_template[template_y_start:template_y_end, template_x_start:template_x_end]

    def visualize_gaussian_map(self, optimal_angle, radius, lidar_samples):
        plt.clf()
        plt.imshow(self.gaussian_map, cmap='hot', interpolation='nearest', origin='upper')  # Correct orientation
        plt.colorbar(label='Intensity')
        plt.title('Gaussian Map with Decay')

        # Plot optimal direction
        angle_rad = np.radians(optimal_angle)
        x_end = self.x_center + radius * np.sin(angle_rad)
        y_end = self.y_center - radius * np.cos(angle_rad)
        plt.plot(x_end, y_end, 'x', color='magenta', markersize=10, label='Optimal Direction')

        # Plot left and right bounds
        left_angle = np.radians(-90)
        x_left = self.x_center + radius * np.sin(left_angle)
        y_left = self.y_center - radius * np.cos(left_angle)
        plt.plot(x_left, y_left, 'x', color='blue', markersize=10, label='Left Bound')

        right_angle = np.radians(90)
        x_right = self.x_center + radius * np.sin(right_angle)
        y_right = self.y_center - radius * np.cos(right_angle)
        plt.plot(x_right, y_right, 'x', color='yellow', markersize=10, label='Right Bound')

        # Plot LiDAR points
        num_samples = len(lidar_samples)
        angles = np.linspace(0, 2 * np.pi, num_samples)
        for i, distance in enumerate(lidar_samples):
            if distance > 0:
                angle = angles[i]
                x = self.x_center + distance * np.sin(angle)
                y = self.y_center - distance * np.cos(angle)
                plt.plot(x, y, 'o', color='cyan', markersize=2, label='LiDAR Point' if i == 0 else "")

        plt.legend()
        plt.pause(0.001)


########################################################################################
# PathPlanner Class
########################################################################################
class PathPlanner:
    def __init__(self, gaussian_map, x_center, y_center):
        self.gaussian_map = gaussian_map
        self.x_center = x_center
        self.y_center = y_center

    def find_optimal_direction(self, radius, gamma=0.99):
        gaussian_map = self.gaussian_map.gaussian_map
        optimal_values = []

        for angle_deg in np.linspace(-90, 90, 360):
            angle_rad = np.radians(angle_deg)
            x_end = self.x_center + radius * np.sin(angle_rad)
            y_end = self.y_center - radius * np.cos(angle_rad)
            x_samples = np.linspace(self.x_center, x_end, 60)
            y_samples = np.linspace(self.y_center, y_end, 60)

            gaussian_values = []
            for idx, (x, y) in enumerate(zip(x_samples, y_samples), start=1):
                if 0 <= int(x) < gaussian_map.shape[1] and 0 <= int(y) < gaussian_map.shape[0]:
                    gaussian_values.append(gaussian_map[int(y), int(x)] * (gamma ** idx))

            total_value = np.sum(np.array(gaussian_values))
            optimal_values.append(total_value)

        optimal_index = np.argmin(optimal_values)
        optimal_direction = np.linspace(-90, 90, 360)[optimal_index]
        return optimal_direction


########################################################################################
# Functions for updating LiDAR and controlling the car
########################################################################################
def update_lidar_and_visualize():
    try:
        lidar_samples = racecar.lidar.get_samples()
        if lidar_samples is not None:

            start = time.time()
            gaussian_map.update_gaussian_map(lidar_samples)
            path_planner = PathPlanner(gaussian_map, gaussian_map.x_center, gaussian_map.y_center)
            radius = 8
            optimal_angle = path_planner.find_optimal_direction(radius)
            control_car(optimal_angle, 0.5)

        gaussian_map.visualize_gaussian_map(optimal_angle, radius, lidar_samples)

    except ValueError as e:
        print(f"Error fetching LiDAR samples: {e}. Skipping this update.")

def control_car(optimal_angle, speed_local):
    global speed, angle
    clipped_angle = np.clip(optimal_angle, -180, 0)
    #angle = normalize(clipped_angle, old_min, old_max, new_min, new_max)
    #speed = speed_local
    print(f"Optimal angle: {optimal_angle}, Speed: {speed_local}")

old_min, old_max = -180, 0
new_min, new_max = -1, 1
def normalize(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

def update():
    global speed, angle
    update_lidar_and_visualize()

########################################################################################
# Main Loop
########################################################################################
if __name__ == "__main__":
    gaussian_map = GaussianMap(sigma=5.3, decay_rate=0.98)
    try:
        racecar.start()
        while True:
            update()
            racecar.set_speed_and_angle(speed, angle)
            time.sleep(0.01)
    except KeyboardInterrupt:
        racecar.close()
