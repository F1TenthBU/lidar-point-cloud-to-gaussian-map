########################################################################################
# Imports
########################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
global speed, angle, speed_offset, angle_offset, accumulated_time

speed = 1.0
angle = 0.0
speed_offset = 0.5
angle_offset = 1.0

accumulated_time = 0  # To accumulate elapsed time

########################################################################################
# Functions from gaussianGenerator.py 
########################################################################################
class GaussianMap:
    def __init__(self, x_res=300, y_res=300, sigma=20):
        self.x_res = x_res
        self.y_res = y_res
        self.sigma = sigma
        self.gaussian_map = np.zeros((x_res, y_res))
        self.x_center = x_res // 2
        self.y_center = y_res // 2

        # Precompute the Gaussian kernel
        size = int(6 * sigma)
        self.half_size = size // 2
        x = np.arange(-self.half_size, self.half_size + 1)
        y = np.arange(-self.half_size, self.half_size + 1)
        xv, yv = np.meshgrid(x, y)
        self.gaussian_kernel = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))

    def update_gaussian_map(self, lidar_samples):
        # Reset the map
        self.gaussian_map = np.zeros((self.x_res, self.y_res))

        num_samples = len(lidar_samples)
        angles = np.linspace(0, 2 * np.pi, num_samples)

        for i, distance in enumerate(lidar_samples):
            if distance == 0:
                continue

            angle = angles[i]
            y = int(self.y_center - distance * np.cos(angle))
            x = int(self.x_center + distance * np.sin(angle))

            if 0 <= x < self.x_res and 0 <= y < self.y_res:
                x_min = max(x - self.half_size, 0)
                x_max = min(x + self.half_size + 1, self.x_res)
                y_min = max(y - self.half_size, 0)
                y_max = min(y + self.half_size + 1, self.y_res)

                kx_min = max(0, self.half_size - x)
                kx_max = kx_min + (x_max - x_min)
                ky_min = max(0, self.half_size - y)
                ky_max = ky_min + (y_max - y_min)

                self.gaussian_map[y_min:y_max, x_min:x_max] += self.gaussian_kernel[ky_min:ky_max, kx_min:kx_max]

    def visualize_gaussian_map(self):
        plt.clf()
        plt.imshow(self.gaussian_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title('Gaussian Map')
        plt.pause(0.001)

########################################################################################
# Define the missing update_lidar_and_visualize() function
########################################################################################

def update_lidar_and_visualize():
    # Obtain LiDAR samples
    lidar_samples = rc.lidar.get_samples()
    if lidar_samples is not None:
        gaussian_map.update_gaussian_map(lidar_samples)
    # Visualize the Gaussian map
    gaussian_map.visualize_gaussian_map()

########################################################################################
# Functions from gaussianMapper.py 
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed, angle, speed_offset, angle_offset, accumulated_time

    speed = 1.0  # The initial speed is at 1.0
    angle = 0.0  # The initial turning angle away from the center is at 0.0
    speed_offset = 0.5  # The initial speed offset is 0.5
    angle_offset = 1.0  # The initial angle offset is 1.0

    accumulated_time = 0  # Reset accumulated time on start

    # This tells the car to begin at a standstill
    rc.drive.stop()

# [FUNCTION] After start() is run, this function is run once every frame
def update():
    global speed, angle, speed_offset, angle_offset, accumulated_time

    # Update the accumulated time with the delta time from each frame
    accumulated_time += rc.get_delta_time()
    
    # Control the RACECAR using the controller
    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0.5:
        speed = speed_offset
    elif rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0.5:
        speed = -speed_offset
    else:
        speed = 0
      
    (x, y) = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    if x > 0.5:
        angle = angle_offset
    elif x < -0.5:
        angle = -angle_offset
    else:
        angle = 0

    if rc.controller.was_pressed(rc.controller.Button.A):
        speed_offset += 0.1
        speed_offset = min(speed_offset, 1.0)
        print(f"Speed Increased! Current Speed Offset: {speed_offset}")
    
    if rc.controller.was_pressed(rc.controller.Button.B):
        speed_offset -= 0.1
        speed_offset = max(speed_offset, 0.0)
        print(f"Speed Decreased! Current Speed Offset: {speed_offset}")

    if rc.controller.was_pressed(rc.controller.Button.X):
        angle_offset += 0.1
        angle_offset = min(angle_offset, 1.0)
        print(f"Angle Increased! Current Angle Offset: {angle_offset}")
    
    if rc.controller.was_pressed(rc.controller.Button.Y):
        angle_offset -= 0.1
        angle_offset = max(angle_offset, 0.0)
        print(f"Angle Decreased! Current Angle Offset: {angle_offset}")

    # Send the speed and angle values to the RACECAR
    rc.drive.set_speed_angle(speed, angle)
    update_lidar_and_visualize()

# def update_slow(): 
#     continue

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":   
    gaussian_map = GaussianMap(sigma=20)   # Increased sigma for wider Gaussians
    plt.ion()  # Enable interactive mode for plotting
    rc.set_start_update(start, update)
    rc.go()
