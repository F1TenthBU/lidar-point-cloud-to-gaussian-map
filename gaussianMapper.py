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
from scipy.interpolate import UnivariateSpline

sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
global speed, angle, speed_offset, angle_offset, accumulated_time, counter
counter=0

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
    def __init__(self, x_res=300, y_res=300, sigma=10, decay_rate=0.99):
        self.x_res = x_res
        self.y_res = y_res
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.gaussian_map = np.zeros((x_res, y_res))
        self.x_center = x_res // 2
        self.y_center = y_res // 2
    
    def update_gaussian_map(self, lidar_samples):
        # Reset the Gaussian map with JAX zeros array
        self.gaussian_map = jnp.zeros((self.x_res, self.y_res))

        num_samples = len(lidar_samples)
        angles = jnp.linspace(0, 2 * jnp.pi, num_samples)

        for i, distance in enumerate(lidar_samples):
            if distance == 0:
                continue

            angle = angles[i]
            y = int(self.y_center - distance * jnp.cos(angle))  # Adjust y-axis
            x = int(self.x_center + distance * jnp.sin(angle))  # Adjust x-axis

            if 0 <= x < self.x_res and 0 <= y < self.y_res:
                xv, yv = jnp.meshgrid(jnp.arange(self.x_res), jnp.arange(self.y_res))
                gaussian = jnp.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * self.sigma ** 2))
                self.gaussian_map += gaussian  # Accumulate Gaussian data

    def visualize_gaussian_map(self):
        plt.clf()  # Clear the previous figure
        plt.imshow(self.gaussian_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title('Gaussian Map with Decay')
        plt.pause(0.001)  # Small pause to update the figure without blocking

def control_racecar(smooth_x, smooth_y, speed=1.0):
    # """
    # Control the car to follow the smooth path curve.
    # """
    # # Example: Calculate the desired steering angle and speed
    # target_x = smooth_x[len(smooth_x) // 2]  # Midpoint of the curve (look-ahead point)
    # steering_angle = np.arctan2(target_x, 1)  # Simple steering calculation

    # # Send speed and steering commands to the car
    # speed = 1.0  # Example constant speed
    # rc.drive.set_speed_angle(speed, steering_angle)
    
    
    # """
    # Control the car to follow the smooth path curve based on the path coordinates.
    # """
    # # Choose two look-ahead points to calculate the slope
    # lookahead_index = min(10, len(smooth_x) - 1)  # Ensure safe index

    # # Calculate the difference between two points (slope)
    # delta_x = smooth_x[lookahead_index] - smooth_x[0]
    # delta_y = smooth_y[lookahead_index] - smooth_y[0]

    # # Calculate the angle in radians
    # raw_steering_angle = np.arctan2(delta_y, delta_x)

    # # Normalize the steering angle to be within [-1.0, 1.0]
    # normalized_steering_angle = np.clip(raw_steering_angle / np.pi, -1.0, 1.0)

    # # Send the speed and normalized steering angle to the racecar
    # speed = 1.0  # Example constant speed
    # rc.drive.set_speed_angle(speed, normalized_steering_angle)

    """
    Calculate the speed and steering angle based on the optimal path.
    """
    # Look-ahead point for smoother steering (adjust index as needed)
    # lookahead_index = min(10, len(smooth_x) - 1)  # Look ahead by 10 steps

    # Calculate the steering angle to the look-ahead point
    delta_x = smooth_x[0] - 150
    delta_y = smooth_y[0] - 150
    steering_angle = np.arctan2(delta_y, delta_x)  # Angle in radians

    # Normalize the steering angle to fit the [-1.0, 1.0] range
    normalized_steering = np.clip(steering_angle / np.pi, -1.0, 1.0)

    # Set a fixed speed (can adjust based on your needs)
    speed = 0.8  # Example speed value (0.8 forward)

    # Send the speed and angle to the car
    rc.drive.set_speed_angle(speed, normalized_steering)
    
def update_lidar_and_visualize():
    try:
        lidar_samples = rc.lidar.get_samples()  # Fetch new lidar data
        print("lidar sample: ", lidar_samples[:6])

        if lidar_samples is not None:
            gaussian_map.update_gaussian_map(lidar_samples)  # Update heatmap

            # Calculate the optimal path
            path_planner = PathPlanner(gaussian_map)
            x, y = path_planner.find_optimal_path()  # Raw path points
            smooth_x, smooth_y = path_planner.fit_curve(x, y)  # Smooth curve

            # control_racecar(smooth_x, smooth_y) # Run the car based on our curve

            # Plot the path on the Gaussian map
            path_planner.plot_path(x, y, smooth_x, smooth_y)

        # gaussian_map.visualize_gaussian_map()  # Display the heatmap

    except ValueError as e:
        print(f"Error fetching LiDAR samples: {e}. Skipping this update.")
    # global accumulated_time

    # lidar_samples = rc.lidar.get_samples()
    # if lidar_samples is not None:
    #     gaussian_map.update_gaussian_map(lidar_samples)

    # # Plot the Gaussian map every UPDATE_INTERVAL
    # if accumulated_time >= UPDATE_INTERVAL:
    #     gaussian_map.visualize_gaussian_map()
    #     accumulated_time = 0  # Reset the accumulated time


########################################################################################
# PathPlanner Class
########################################################################################
class PathPlanner:
    def __init__(self, gaussian_map):
        self.gaussian_map = gaussian_map  # 2D Gaussian heatmap

    def find_optimal_path(self):
        """
        Find the path through the lowest Gaussian values, considering dynamic bounds.
        """
        heatmap = self.gaussian_map.gaussian_map
        y_res, x_res = heatmap.shape
        car_position = (150,150)

        # Set dynamic boundaries based on the car's position (example logic)
        min_x_bound = max(0, car_position[0] - 100)
        max_x_bound = min(x_res, car_position[0] + 100)

        min_y_bound = max(0, car_position[1] - 100)
        max_y_bound = min(y_res, car_position[1] + 100)

        optimal_x = []
        y_coords = np.arange(min_y_bound, max_y_bound)

        for y in y_coords:
            row = heatmap[y, min_x_bound:max_x_bound]
            min_x = np.argmin(row) + min_x_bound  # Shift index to original range
            optimal_x.append(min_x)

        return np.array(optimal_x), y_coords

    # Last
    # def find_optimal_path(self):
    #     """
    #     Find the path through the lowest Gaussian values within track boundaries.
    #     """
    #     heatmap = self.gaussian_map.gaussian_map
    #     y_res, x_res = heatmap.shape

    #     optimal_x = []
    #     y_coords = np.arange(y_res)

    #     # Restrict the search within a valid x-range (track boundary)
    #     min_x_bound, max_x_bound = 50, 250  # Example boundaries

    #     for y in range(y_res):
    #         row = heatmap[y, min_x_bound:max_x_bound]
    #         min_x = np.argmin(row) + min_x_bound  # Shift index to original range
    #         optimal_x.append(min_x)

    #     return np.array(optimal_x), y_coords

    # def find_optimal_path(self):
    #     """
    #     Find the path through the lowest Gaussian values.
    #     Returns the x and y coordinates of the optimal path.
    #     """
    #     heatmap = self.gaussian_map.gaussian_map  # 2D map of Gaussian values
    #     y_res, x_res = heatmap.shape  # Get the resolution of the map

    #     # Find the index of the minimum value in each row (or along the y-axis)
    #     optimal_x = [np.argmin(heatmap[y, :]) for y in range(y_res)]

    #     # Create corresponding y coordinates (rows)
    #     y_coords = np.arange(y_res)

    #     # Return the coordinates as (x, y)
    #     return np.array(optimal_x), y_coords

    def fit_curve(self, x, y):
        """
        Fit a smooth curve through the (x, y) points using spline interpolation.
        """
        # Fit a spline to the path for smoothness
        spline = UnivariateSpline(y, x, s=1)  # s is the smoothing factor
        smooth_y = np.linspace(y[0], y[-1], 500)  # Smooth y values
        smooth_x = spline(smooth_y)  # Smooth x values

        return smooth_x, smooth_y

    def plot_path(self, x, y, smooth_x, smooth_y):
        """
        Plot the Gaussian map and the calculated path.
        """
        plt.clf()  # Clear the previous figure

        # Plot the Gaussian heatmap
        plt.imshow(self.gaussian_map.gaussian_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')  # Add a colorbar for reference

        # Plot the raw points and the smooth path curve
        plt.plot(x, y, 'bo', markersize=5, label='Optimal Points')  # Raw points
        plt.plot(smooth_x, smooth_y, 'r-', linewidth=2, label='Smooth Path')  # Smooth path

        # Plot the starting point (smooth_x[0], smooth_y[0]) as an 'X'
        plt.plot(150, 150, 'kx', markersize=10, markeredgewidth=2, label='Car', color='hotpink')
        plt.plot(smooth_x[0], smooth_y[0], 'kx', markersize=10, markeredgewidth=2, label='Car', color='yellow')


        # Add labels and a legend
        plt.title('Optimal Path on Gaussian Heatmap')
        plt.legend(loc='upper right')

        # Pause briefly to update the plot without blocking
        plt.pause(0.001)



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

    accumulated_time = 0.0  # Reset accumulated time on start

    # This tells the car to begin at a standstill
    rc.drive.stop()

# [FUNCTION] After start() is run, this function is run once every frame, 60fps
def update():
    global speed, angle, speed_offset, angle_offset, accumulated_time, counter

    # Update the accumulated time with the delta time from each frame
    accumulated_time += rc.get_delta_time()
    counter+=1
    print("counter: ", counter)
    print("time: ", accumulated_time)
    # print("time before if",  accumulated_time)
    
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
    
# def update_slow(): 
#     update_lidar_and_visualize()


#set a maximum and make the std larger. Keep the last 360 gaussians in a buffer. Delete the rest.
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":   
    gaussian_map = GaussianMap(sigma=8, decay_rate=0.98)   
    plt.ion()  # Enable interactive mode for plotting
    rc.set_start_update(start, update)
    # rc.set_start_update(start, update, update_slow)
    # rc.set_update_slow_time(1/6)
    rc.go()