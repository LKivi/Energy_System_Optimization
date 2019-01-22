# -*- coding: utf-8 -*-
"""

Author: Marco Wirtz, Institute for Energy Efficient Buildings and Indoor Climate, RWTH Aachen University, Germany

Created: 01.09.2018

"""


import os
import datetime
import numpy as np
import clustering_medoid as clustering
import time

part_load = 1  # 1: partload behaviour enabled, 0: partload behaviour not considered

start_time = time.time()
  
# Objective function
# available: "tac", "co2_onsite", "co2_net", "co2_gross", "invest", "power_from_grid", "net_power_from_grid", "renewable_abs"
obj_1 = "tac"          # First objective function
obj_2 = "co2_gross"    # Second objective function

# Number of pareto points for each objective function in epsilon constraint procedure
pareto_points = 4

# Create result directory
dir_results = str(os.path.dirname(os.path.realpath(__file__))) + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_multi-objective__" + obj_1 + "__" + obj_2)

# Set path to input files
#    path_weather_file = "input_data/CHN_Shanghai.Shanghai.583670_IWEC.csv"
path_heating_load = "input_data/heating.csv"
path_cooling_load = "input_data/cooling.csv"
path_power_load   = "input_data/electicity.csv"

demands = {}

# load time series as numpy array
demands["heat"] = np.loadtxt(open(path_heating_load, "rb"), delimiter=",", usecols=(1))        # MW, heating load
demands["cool"] = np.loadtxt(open(path_cooling_load, "rb"), delimiter=",", usecols=(1))        # MW, cooling load
demands["power"] = np.loadtxt(open(path_power_load, "rb"), delimiter=",", usecols=(1))         # MW, electrical load grid

# Improve numeric by deleting very small loads
eps = 0.01 # MW
for load in ["power", "heat", "cool"]:
    for k in range(len(demands[load])):
        if demands[load][k] < eps:
            demands[load][k] = 0

#%% CLUSTERING

number_clusters = 15
inputs_clustering = np.array([demands["heat"],
                              demands["power"],
                              demands["cool"]])
                                                               
(inputs, nc, z) = clustering.cluster(inputs_clustering, 
                                     number_clusters,
                                     norm = 2,
                                     mip_gap = 0.01,
                                     weights = [1,1,1])
             
# Determine time steps per day
len_day = int(inputs_clustering.shape[1] / 365)

clustered = {}
clustered["heat"]        = inputs[0]
clustered["power"]       = inputs[1]
clustered["cool"]        = inputs[2]

clustered["weights"]     = nc
clustered["days"]        = number_clusters
clustered["time_steps"]  = 24


#%% Multi-objective procedure    

## Find single objetive optimum
#anchor_point_1 = {}
#anchor_point_2 = {}
#all_sol = []
#
## Calculate anchor point for first objective function
## 1.) Single-objective optimization for first objective function
#opt_res = run_optim(obj_1, "", "", str(dir_results + "\\single_objective_" + obj_1), clustered, part_load)
#
## 2.) Single-objective optimization for second objective function, in which the first objective function is bounded to its optimum from the previous optimization
#anchor_point_1 = run_optim(obj_2, obj_1, opt_res[obj_1], str(dir_results + "\\anchor_point_" + obj_1), clustered, part_load)
#
## Append this solution to the solution list
#all_sol.append(anchor_point_1)
#
## Print result for first anchor point
#print("\nCalculated first anchor point for " + obj_1)
#print(">> " + obj_1 + ": " + str(anchor_point_1[obj_1]))
#print(obj_2 + ": " + str(anchor_point_1[obj_2]))
#
## Calculate anchor point for second objective function
## 1.) Single-objective optimization for second objective function
#opt_res = run_optim(obj_2, "", "", str(dir_results + "\\single_objective_" + obj_2), clustered, part_load)
#
## 2.) Single-objective optimization for first objective function, in which the second objective function is bounded to its optimum from the previous optimization
#anchor_point_2 = run_optim(obj_1, obj_2, opt_res[obj_2], str(dir_results + "\\anchor_point_" + obj_2), clustered, part_load)
#
## Append this solution to the solution list
#all_sol.append(anchor_point_2)
#
## Print results
#print("\nCalculated second anchor point for " + obj_2)
#print(obj_1 + ": " + str(anchor_point_1[obj_1]))
#print(">> " + obj_2 + ": " + str(anchor_point_2[obj_2]))
#
#print("\nAnchor point: " + obj_1 + " | " + obj_2)
#print("Anchor point 1 (optimal " + obj_1 + "): >>" + str(round(anchor_point_1[obj_1])) + "<< | " + str(round(anchor_point_1[obj_2])))
#print("Anchor point 2 (optimal " + obj_2 + "): " + str(round(anchor_point_2[obj_1])) + " | >>" + str(round(anchor_point_2[obj_2])) + "<<")
#
## Run epsilon constraint procedure
#for obj in [obj_1, obj_2]:
#    if obj == obj_1:
#        obj_min = obj_1 # objective function to be minimized
#        obj_eps = obj_2 # objective function bounded by epsilon
#    elif obj == obj_2:
#        obj_min = obj_2 # objective function to be minimized
#        obj_eps = obj_1 # objective function bounded by epsilon
#    print("\nCalculate pareto points (objective function: " + obj_min + ")")
#    
#    # Calculate epsilon constraints
#    delta_eps = (anchor_point_1[obj_eps] - anchor_point_2[obj_eps])/(pareto_points + 1)
#    eps_constr = [(anchor_point_2[obj_eps] + k * delta_eps) for k in range(1, pareto_points + 1)]
#    print("Calculated epsilon constraints: " + str(eps_constr))
#    
#    # Calculate pareto points
#    for eps in range(len(eps_constr)):
#        all_sol.append(run_optim(obj_min, obj_eps, eps_constr[eps], str(dir_results + "\\pareto_" + str(len(all_sol)-1)), clustered, part_load))
#
#print("Multi-objective optimization finished. (%f seconds.)" %(time.time() - start_time))
#
##%% POST PROCESSING
#import post_processing_run
#post_processing_run.run_post_processing(dir_results, clustered)

   

    