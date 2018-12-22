# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:23:57 2018

@author: mwi
"""

import numpy as np
import bldg_balancing_heur as heur
import bldg_balancing_optim as opt
import bldg_balancing_optim_all_nodes as opt_all


# Calculate mass flow through building (intra-balancing): "> 0": flow from supply to return pipe   

def design_buildings(nodes, param, devs_dom, dir_results):
    
    dir_buildings = dir_results + "\\buildings"

    if not param["switch_building_optimization"]:
        nodes, param = heur.calc_residuals(nodes, param, devs_dom, dir_buildings)
    
    else:   
        
        if param["switch_nodewise"]:
            
            for n in range(len(nodes)):
                dir_node = dir_buildings + "\\" + str(nodes[n]["name"])
                nodes[n], param = opt.run_optim(nodes[n], param, devs_dom, dir_node)
        
        else:
            
            nodes, param = opt_all.run_optim(nodes, param, devs_dom, dir_buildings)
            
            
    
        
    residual = get_residual(nodes)
    
    
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
    