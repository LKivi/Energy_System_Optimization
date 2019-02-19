# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 

@author: lkivi
"""


import os

import parameters
import device_optim

import datetime
import numpy as np
import time

import matplotlib.pyplot as plt

# Choose use case
#use_case = "FZJ"
use_case = "DOC_plots"

# Choose scenario
 
#scenario = "stand_alone"                     # stand-alone supply
#scenario = "conventional_DHC"                # conventional, separated heating and cooling network
#scenario = "Ectogrid_min"                    # bidirectional network with conventional BU devices and minumum building equipment
#scenario = "Ectogrid_full"                   # bidirectional network with full BU & building equipment


N = 10

DOC_list = np.zeros(N+1)
tac_list = np.zeros(N+1)

for i in range(N+1):
    
    shift = 4380/N * i

    # Define paths
    path_file = str(os.path.dirname(os.path.realpath(__file__)))
    dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + "_" + scenario
    
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)


    ## Load parameters
    nodes, param, devs, devs_dom = parameters.load_params(use_case, path_file, scenario, shift)
    param["switch_post_processing"] = 0
    
    # Run device optimization
    nodes, param = device_optim.run(nodes, param, devs, devs_dom, dir_results)
    
    
    # Calculate and store KPIs
    
    # Read solution file
    file_name = dir_results + "\\model.sol"
    with open(file_name, "r") as solution_file:
        file = solution_file.readlines()  
            
    time_steps = range(24)
    n_days = param["n_clusters"]
    
    # DOC
    DOC =  2 * sum(sum( min(nodes[0]["heat"][d][t], nodes[1]["cool"][d][t]) for t in time_steps) * param["day_weights"][d] for d in range(n_days)) / sum(sum(nodes[0]["heat"][d][t] + nodes[1]["cool"][d][t] for t in time_steps) * param["day_weights"][d] for d in range(n_days))     
    DOC_list[i] = DOC
    
    # tac
    for line in range(len(file)):
        if "total_annualized_costs" in file[line]:
            tac = float(str.split(file[line])[1])
            break   
    tac_list[i] = tac
    
    

plt.plot(DOC_list, tac_list)
plt.xlabel("DOC")
plt.ylabel("tac")
    
    
    
    
    
    