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

########################################################################################
# Functions from gaussianGenerator.py 
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
        self.gaussian_map *= self.decay_rate
        num_samples = len(lidar_samples)
        angles = np.linspace(0, 2 * np.pi, num_samples)

        for i, distance in enumerate(lidar_samples):
            if distance == 0:
                continue

            angle = angles[i]
            y = int(self.y_center - distance * np.cos(angle))  # Adjust y-axis for correct alignment
            x = int(self.x_center + distance * np.sin(angle))  # Adjust x-axis for lateral placement

            if 0 <= x < self.x_res and 0 <= y < self.y_res:
                xv, yv = np.meshgrid(np.arange(self.x_res), np.arange(self.y_res))
                gaussian = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * self.sigma ** 2))
                self.gaussian_map += gaussian

    def visualize_gaussian_map(self):
        plt.clf()  # Clear the previous figure
        plt.imshow(self.gaussian_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Intensity')
        plt.title('Gaussian Map with Decay')
        plt.pause(0.001)  # Small pause to update the figure without blocking

def update_lidar_and_visualize():
    global accumulated_time

    lidar_samples = rc.lidar.get_samples()
    if lidar_samples is not None:
        gaussian_map.update_gaussian_map(lidar_samples)

    # Plot the Gaussian map every second (1.0 seconds)
    if accumulated_time >= 1.0:
        gaussian_map.visualize_gaussian_map()
        accumulated_time = 0  # Reset the accumulated time

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

def update_slow(): 
    update_lidar_and_visualize()


#set a maximum and make the std larger. Keep the last 360 gaussians in a buffer. Delete the rest.
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":   
    gaussian_map = GaussianMap(sigma=8, decay_rate=0.98)   
    plt.ion()  # Enable interactive mode for plotting
    rc.set_start_update(start, update, update_slow)
    rc.go()