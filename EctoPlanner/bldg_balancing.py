# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:23:57 2018

@author: mwi
"""

import numpy as np

 # Calculate mass flow through building (intra-balancing): "> 0": flow from supply to return pipe   
      
def calc_residuals(nodes, param, time_steps, dir_results):
    for n in range(len(nodes)):
        
        # Determine capacity of heat pump, electric heater
        if param["use_eh_in_bldgs"] == 1:
            heat_dem_sort = np.flip(np.sort(nodes[n]["heat"]),0)
            hp_capacity = heat_dem_sort[param["op_hours_el_heater"]] #1000 hrs
            eh_capacity = heat_dem_sort[0] - hp_capacity
        
        elif param["use_eh_in_bldgs"] == 0:
            hp_capacity = np.max(nodes[n]["heat"])
            eh_capacity = 0
        
        power = {"HP": np.zeros(len(time_steps)), 
                 "CC": np.zeros(len(time_steps)),
                 "EH": np.zeros(len(time_steps)),
                 }
        res_heat_dem = np.zeros(len(time_steps))
        
        for t in time_steps:
            if nodes[n]["heat"][t] <= hp_capacity:
                power["HP"][t] = nodes[n]["heat"][t] / param["COP_HP"]
                power["EH"][t] = 0
            elif nodes[n]["heat"][t] > hp_capacity:
                power["HP"][t] = hp_capacity / param["COP_HP"]
                power["EH"][t] = (nodes[n]["heat"][t] - hp_capacity) / param["eta_th_eh"]
                
            power["CC"][t] = nodes[n]["cool"][t] / param["COP_CC"]       
            res_heat_dem[t] = ((nodes[n]["heat"][t] - (power["EH"][t] * param["eta_th_eh"])) - power["HP"][t]) - (power["CC"][t] + nodes[n]["cool"][t])

        mass_flow = res_heat_dem * 1000 / (param["c_f"] * (param["T_hot"] - param["T_cold"]))

        nodes[n]["hp_capacity"] = hp_capacity # kW_th
        nodes[n]["eh_capacity"] = eh_capacity # kW_th
        nodes[n]["cc_capacity"] = np.max(nodes[n]["cool"]) # kW_th
        nodes[n]["res_heat_dem"] = res_heat_dem # kW_th
        nodes[n]["mass_flow"] = mass_flow # from supply to return pipe, kg/s

        nodes[n]["power_HP"] = power["HP"]
        nodes[n]["power_EH"] = power["EH"]
        nodes[n]["power_CC"] = power["CC"]
        
#        heat_from_net_no_balancing = sum(param["COP_HP"][t] * power["HP"][t] - power["HP"][t] for t in time_steps)
#        cool_from_net_no_balancing = sum(param["COP_CC"][t] * power["CC"][t] + power["CC"][t] for t in time_steps)
#        sum_res_heat_dem = sum(res_heat_dem[t] for t in time_steps)
        
    return nodes