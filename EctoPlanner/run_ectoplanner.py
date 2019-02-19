# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:07:27 2018

@author: mwi
"""

import os

import parameters
import device_optim
import network

import datetime
import numpy as np
import time

#%%


# Choose use case
use_case = "FZJ"
#use_case = "DOC_plots"

# Choose scenario
 
#scenario = "stand_alone"                     # stand-alone supply
#scenario = "conventional_DHC"                # conventional, separated heating and cooling network
#scenario = "Ectogrid_min"                    # bidirectional network with conventional BU devices and minumum building equipment
scenario = "Ectogrid_full"                   # bidirectional network with full BU & building equipment




#scenario = "typedays_1bal"                              # Typedays, single thermal balance for BU design, TES in buildings
#scenario = "typedays_absC"                              # Clustering, seperated heating and cooling balance for BU design, TES in buildings
#scenario = "typedays_1bal_noBldgTES"                    # Clustering, single thermal balance for BU design, no building storages
#scenario = "typedays_absC_noBldgTES"                    # Clustering, seperated heating and cooling balance for BU design, no building storages
#scenario = "typedays_standalone"
#scenario = "1bal_noBldgTES"



# Define paths
path_file = str(os.path.dirname(os.path.realpath(__file__)))
dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + "_" + scenario

if not os.path.exists(dir_results):
    os.makedirs(dir_results)
#
#
## Load parameters
nodes, param, devs, devs_dom = parameters.load_params(use_case, path_file, scenario)





# Run device optimization
nodes, param = device_optim.run(nodes, param, devs, devs_dom, dir_results)

    
           
# Optimize network topology
#param = network.design_network(nodes, param, dir_results)






# Calculate inter-balancing and design balancing unit
#balancing_unit.design_balancing_unit(nodes, devs, param, residual, dir_results)





# Post-processing
#print("\n-----Post-processing:-----")
#post_processing.calc_diversity_index(nodes, time_steps)
#post_processing.plot_residual_heat(dir_results)
#post_processing.plot_total_power_dem_bldgs(dir_results)
#post_processing.plot_power_dem_HP_EH_CC(dir_results)
#post_processing.plot_demands(nodes, dir_results)
#post_processing.plot_COP_HP_CC(param, dir_results)

#        post_processing.plot_ordered_load_curve(heat_dem_sort, hp_capacity, eh_capacity, param, nodes[n]["name"], time_steps, dir_results)
#        post_processing.plot_bldg_balancing(nodes[n], time_steps, param, dir_results)
#        post_processing.save_balancing_results(nodes[n], time_steps, dir_results)