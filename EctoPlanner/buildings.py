# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:23:57 2018

@author: mwi
"""

import numpy as np
import bldg_balancing_heur as heur
import bldg_balancing_optim_nodewise as opt
import bldg_balancing_optim_complete as opt_compl
import bldg_balancing_optim_complete_clustered as opt_compl_clustered


# Calculate mass flow through building (intra-balancing): "> 0": flow from supply to return pipe   

def design_buildings(nodes, param, devs, devs_dom, dir_results):
    

            
    if param["switch_clustering"]:
        nodes, param = opt_compl_clustered.run(nodes, param, devs, devs_dom, dir_results)
    else:
        nodes, param = opt_compl.run(nodes, param, devs, devs_dom, dir_results)
            
            
            
    
    residual = 0 
#    residual = get_residual(nodes)
    
    
    return nodes, residual









#%%

# Get total residual load to be provided by the balancing unit
def get_residual(nodes):
    
    time_steps = range(8760)
    
    sum_residual_heat = np.zeros(8760)
    sum_power_dem_bldgs = np.zeros(8760)
    for t in time_steps:
        sum_residual_heat[t] = sum(nodes[n]["res_heat_dem"][t] for n in range(len(nodes)))
        sum_power_dem_bldgs[t] = sum(nodes[n]["power_dem"][t] for n in range(len(nodes))) 
    

    # Network residual loads
    residual = {}
    residual["heat"] = np.zeros(8760)
    residual["cool"] = np.zeros(8760)
    for t in time_steps:
        if sum_residual_heat[t] > 0:
            residual["heat"][t] = sum_residual_heat[t] / 1000           # MW, total residual heat demand
        else:
            residual["cool"][t] = (-1) * sum_residual_heat[t] / 1000    # MW, total residual cooling demand
    residual["power"] = sum_power_dem_bldgs / 1000                      # MW, total electricity demand for devices in buildings
    
    
    return residual
    