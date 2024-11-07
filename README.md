The Lidar Point Cloud to Gaussian Map script (gaussianGenerator.py) processes lidar point cloud data to generate Gaussian maps that represent obstacles in the environment. Currently, the autonomous car in the simulation is not driving, so the script creates a static Gaussian map based on the lidar data captured at the moment when the script is run. Each obstacle is represented as a Gaussian distribution, capturing its relative position and size. Similar to a heat map, yellow indicates higher intensity or risk due to close proximity to the wall.

You can run this script as you do with all other scripts that we run on the simulation. Make sure that the simulation is opened and running, to run the script, navigate to the working directory of your script and type the command "racecar sim gaussianMapper_radius.py". After this, just hit return (on Mac) or the corresponding command to drive autonomously via python script on the simulation. 
