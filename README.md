# gaussianGenerator.py
The Lidar Point Cloud to Gaussian Map script (`gaussianGenerator.py`) processes lidar point cloud data to generate Gaussian maps that represent obstacles in the environment. Currently, the autonomous car in the simulation is not driving, so the script creates a static Gaussian map based on the lidar data captured at the moment when the script is run. Each obstacle is represented as a Gaussian distribution, capturing its relative position and size. Similar to a heat map, yellow indicates higher intensity or risk due to close proximity to the wall.

You can run this script as you do with all other scripts that we run on the simulation. Make sure that the simulation is opened and running, to run the script, navigate to the working directory of your script and type the command "racecar sim gaussianGenerator.py". After this, just hit return (on Mac) or the corresponding command to drive autonomously via python script on the simulation. 

# gaussianMapper_radius.py
This script finds the optimal path by following logic:
1. Analyzing gaussian data for front half-circle of lidar. It takes $\frac{lidar \ datas}{2}$ points on the front half-circle, then connect the points to the lidar center to form a line.
2. We assign 50 points to each connected line with uniform distance, then grab the gaussian value at each point. For each line, we collect the point with maximum value.
3. With $\frac{lidar \ datas}{2}$ maximum points in total, we find the minimum of them all, and declare this point as our "heading point"
4. Control the car based on this calculation and provide the car with angle and speed.

You can run this script as you do with all other scripts that we run on the simulation. Make sure that the simulation is opened and running, to run the script, navigate to the working directory of your script and type the command "racecar sim gaussianMapper_radius.py". After this, just hit return (on Mac) or the corresponding command to drive autonomously via python script on the simulation. 
