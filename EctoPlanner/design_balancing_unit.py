# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:23:57 2018

@author: mwi
"""

import numpy as np
import optim_model

 # Calculate mass flow through balancing unit (inter-balancing): "> 0": flow from supply to return pipe         
def design_balancing_unit(nodes, devs, param, time_steps, dir_results):

    sum_residual_heat = sum(nodes[n]["res_heat_dem"] for n in range(len(nodes)))
    sum_power_dem_bldgs = sum(nodes[n]["power_HP"] + nodes[n]["power_EH"] + nodes[n]["power_CC"] for n in range(len(nodes)))  
    sum_power_dem_HP = sum(nodes[n]["power_HP"] for n in range(len(nodes)))  
    sum_power_dem_EH = sum(nodes[n]["power_EH"] for n in range(len(nodes)))  
    sum_power_dem_CC = sum(nodes[n]["power_CC"] for n in range(len(nodes)))  


    # Write total residual heat in file
    fo = open(dir_results + "\\sum_residual_heat.txt", "w")
    for t in time_steps:
        fo.write(str(round(sum_residual_heat[t],8)) + str("\n"))    

    # Write total power demand of buildings (HP, CC, EH) in file
    fo = open(dir_results + "\\sum_power_dem_bldgs.txt", "w")
    for t in time_steps:
        fo.write(str(round(sum_power_dem_bldgs[t],8)) + str("\n"))    

    # Write total power demand of buildings HP in file
    fo = open(dir_results + "\\sum_power_dem_HP.txt", "w")
    for t in time_steps:
        fo.write(str(round(sum_power_dem_HP[t],8)) + str("\n"))

    # Write total power demand of buildings EH in file
    fo = open(dir_results + "\\sum_power_dem_EH.txt", "w")
    for t in time_steps:
        fo.write(str(round(sum_power_dem_EH[t],8)) + str("\n"))

    # Write total power demand of buildings CC in file
    fo = open(dir_results + "\\sum_power_dem_CC.txt", "w")
    for t in time_steps:
        fo.write(str(round(sum_power_dem_CC[t],8)) + str("\n"))

    
    residual = {}
    residual["heat"] = np.zeros(8760)
    residual["cool"] = np.zeros(8760)
    for t in time_steps:
        if sum_residual_heat[t] > 0:
            residual["heat"][t] = sum_residual_heat[t]
        else:
            residual["cool"][t] = (-1) * sum_residual_heat[t]
    residual["power"] = sum_power_dem_bldgs
        
    res_obj = optim_model.run_optim(devs, param, residual, time_steps, dir_results)
    print(res_obj)

    return nodes