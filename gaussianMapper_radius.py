"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-outreach-labs

File Name: lab_c.py
Title: Lab C - RACECAR Controller

Author: Christopher Lai (MITLL)

Purpose: Using a Python script and the data polled from the controller module,
write code to replicate a manual control scheme for the RACECAR. Gain a mastery
in using conditional statements, controller functions and an understanding in the
rc.drive.set_speed_angle() function. Complete the lines of code under the #TODO indicators 
to complete the lab.

Expected Outcome: When the user runs the script, they are able to control the RACECAR
using the following keys:
- When the right trigger is pressed, the RACECAR drives forward
- When the left trigger is pressed, the RACECAR drives backward
- When the left joystick's x-axis has a value of greater than 0, the RACECAR's wheels turns to the right
- When the value of the left joystick's x-axis is less than 0, the RACECAR's wheels turns to the left
- When the "A" button is pressed, increase the speed and print the current speed to the terminal window
- When the "B" button is pressed, reduce the speed and print the current speed to the terminal window
- When the "X" button is pressed, increase the turning angle and print the current turning angle to the terminal window
- When the "Y" button is pressed, reduce the turning angle and print the current turning angle to the terminal window
"""

########################################################################################
# Imports
########################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

sys.path.insert(1, '../../library')
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

# Lidar parameters
LIDAR_FREQUENCY = 6  # 6 Hz (6 updates per second), according to current sim
UPDATE_INTERVAL = 1.0 / LIDAR_FREQUENCY  # ~0.167 seconds per update

########################################################################################
# GaussianMap Class
########################################################################################
class GaussianMap:
    def __init__(self, x_res=100, y_res=100, sigma=1.0/30.0, decay_rate=0.99):
        self.x_res = x_res
        self.y_res = y_res
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.gaussian_map = np.zeros((x_res, y_res))
        self.world_width = 1.0
        self.world_height = 1.0
        self.x_center = self.world_width / 2
        self.y_center = self.world_height / 2
    
    def update_gaussian_map(self, lidar_samples):
        # Reset the Gaussian map with JAX zeros array
        self.gaussian_map = np.zeros((self.x_res, self.y_res))

        num_samples = len(lidar_samples)
        angles = np.linspace(0, 2 * np.pi, num_samples)

        for i, distance in enumerate(lidar_samples):
            if distance == 0:
                continue
            distance *= 1/300.0
            angle = angles[i]
            y = self.y_center - distance * np.cos(angle)  # Adjust y-axis
            x = self.x_center + distance * np.sin(angle)  # Adjust x-axis

            if 0 <= x < self.x_res and 0 <= y < self.y_res:
                xv, yv = np.meshgrid(np.linspace(0, self.world_width, self.x_res), np.linspace(0, self.world_height, self.y_res))
                gaussian = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * self.sigma ** 2))
                self.gaussian_map += gaussian  # Accumulate Gaussian data

    def visualize_gaussian_map(self, optimal_angle, radius):
        plt.clf()  # Clear the previous figure
        plt.imshow(self.gaussian_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title('Gaussian Map with Decay')

        angle_rad = np.radians(optimal_angle)
        x_end = self.x_center - radius * np.cos(angle_rad)
        y_end = self.y_center + radius * np.sin(angle_rad)
        print("optimal coord: ", x_end, y_end)
        plt.plot(x_end *100, y_end *100, 'x', color='magenta', markersize=10, label='Optimal Direction')

        plt.pause(0.001)  # Small pause to update the figure without blocking

########################################################################################
# PathPlanner Class
########################################################################################
class PathPlanner:
    def __init__(self, gaussian_map, x_center, y_center):
        self.gaussian_map = gaussian_map  # 2D Gaussian heatmap
        self.x_center = x_center
        self.y_center = y_center

    def find_optimal_direction(self, radius, gamma=0.96):
        """
        Find the optimal direction to go based on the Gaussian map within a half-circle.
        Parameters:
            x_center, y_center: Center position of the car.
            radius: Radius of the half-circle in front of the car.
        Returns:
            optimal_direction: Angle (in degrees) of the optimal direction to go.
        """
        gaussian_map = self.gaussian_map.gaussian_map  # Assuming a 2D array of Gaussian values
        optimal_values = []  # To store summed Gaussian values for each angle
        min_value_points = [] 
        
        # Iterate over the 180-degree half-circle in front (360 points for finer resolution)
        for angle_deg in np.linspace(-90, 90, 360):  # -90 to 90 degrees relative to the car's heading
            angle_rad = np.radians(angle_deg)
            
            # Calculate the end point of the line on the circle
            x_end = self.x_center + radius * np.cos(angle_rad)
            y_end = self.y_center + radius * np.sin(angle_rad)
            
            # Sample 50 points from the end point to the center
            x_samples = np.linspace(self.x_center, x_end, 50)
            y_samples = np.linspace(self.y_center, y_end, 50)
            
            # Get the Gaussian values at each sampled point
            gaussian_values = []
            for idx, (x, y) in enumerate(zip(x_samples, y_samples), start=1):
                # Ensure the points are within bounds of the gaussian_map
                # if idx % 10 == 0:
                #     breakpoint()
                if 0 <= x < self.gaussian_map.world_width and 0 <= y < self.gaussian_map.world_height:
                    weighted_value = gaussian_map[int(y*self.gaussian_map.y_res), int(x*self.gaussian_map.x_res)] * (gamma ** idx)
                    gaussian_values.append(weighted_value)
                    # gaussian_values.append((weighted_value, int(x), int(y)))
            
            # # Get minimum Gaussian values for this direction
            minimum_value = np.sum(gaussian_values)
            optimal_values.append(minimum_value)
    

        # Find the direction with the minimum Gaussian value
        optimal_index = np.argmin(optimal_values)
        optimal_direction = np.linspace(-90, 90, 360)[optimal_index]
        
        return optimal_direction


########################################################################################
# Functions for update lidar and controlling the car
########################################################################################
def control_car(optimal_angle, speed):
    """
    Control the car based on the optimal direction calculated from LiDAR data.
    """
    # Calculate the steering angle based on the optimal direction
    # Assuming `optimal_angle` is the angle in radians to turn towards
    # Normalize the angle to fit within -1.0 (left) and 1.0 (right) for the steering
    max_turn_angle = np.pi / 4  # 45 degrees, adjust as needed
    steering_angle = np.clip(optimal_angle / max_turn_angle, -1.0, 1.0)

    # Send the control command to the car
    rc.drive.set_speed_angle(speed, steering_angle)

import time
def update_lidar_and_visualize():
    try:
        lidar_samples = rc.lidar.get_samples()  # Fetch new lidar data
        # print("lidar sample: ", lidar_samples[:6])

        if lidar_samples is not None:
            start = time.time()
            gaussian_map.update_gaussian_map(lidar_samples)  # Update heatmap
            # end = rc.get_delta_time()
            print("accumulated_time:, ", time.time() - start)

            # Calculate the optimal path
            path_planner = PathPlanner(gaussian_map, gaussian_map.x_center, gaussian_map.y_center)
            radius = 5.0/30.0
            speed = 1
            optimal_angle = path_planner.find_optimal_direction(radius)
            print("optimal_angle: ", optimal_angle)
            control_car(optimal_angle, speed)


        # gaussian_map.visualize_gaussian_map(optimal_angle, radius)  # Display the heatmap

    except ValueError as e:
        print(f"Error fetching LiDAR samples: {e}. Skipping this update.")


########################################################################################
# Functions for racecar: start(), update()
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed, angle, speed_offset, angle_offset, accumulated_time

    speed = 1.0  # The initial speed is at 1.0
    angle = 0.0  # The initial turning angle away from the center is at 0.0
    speed_offset = 0.5  # The initial speed offset is 0.5
    angle_offset = 1.0  # The initial angle offset is 1.0

    accumulated_time = 0.0  # Reset accumulated time on start

    # This tells the car to begin at a standstill
    rc.drive.stop()

# [FUNCTION] After start() is run, this function is run once every frame, 60fps
def update():
    global speed, angle, speed_offset, angle_offset, accumulated_time

    # Update the accumulated time with the delta time from each frame
    accumulated_time += rc.get_delta_time()
    # print("time: ", accumulated_time)
    
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

    if accumulated_time >= UPDATE_INTERVAL:
        # print("time",  accumulated_time)
        update_lidar_and_visualize()
        accumulated_time = 0
    


#set a maximum and make the std larger. Keep the last 360 gaussians in a buffer. Delete the rest.
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":   
    gaussian_map = GaussianMap(sigma=8.0/300.0, decay_rate=0.98)   
    plt.ion()  # Enable interactive mode for plotting
    rc.set_start_update(start, update)
    rc.go()