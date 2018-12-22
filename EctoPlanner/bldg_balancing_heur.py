# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:17:36 2018

@author: lkivi
"""

import numpy as np

# Heuristic calculation of residual loads      
def calc_residuals(nodes, param, devs_dom, dir_results):
    
    print("HIER FEHLT NOCH BERECHNUNG TAC_TBE")
    print("AIR COOLING FEHLT NOCH")
        
    
    time_steps = range(8760)
    
    for n in range(len(nodes)):
        
        # Determine capacity of heat pump and electric heater
        if param["use_eh_in_bldgs"] == 1:
            heat_dem_sort = np.flip(np.sort(nodes[n]["heat"]),0)
            hp_capacity = heat_dem_sort[param["op_hours_el_heater"]]   # Operating hours of electric heater are specified in params
            eh_capacity = heat_dem_sort[0] - hp_capacity
        
        elif param["use_eh_in_bldgs"] == 0:
            hp_capacity = np.max(nodes[n]["heat"])
            eh_capacity = 0
        
       
        # initialize time series
        power = {"HP": np.zeros(len(time_steps)), 
                 "CC": np.zeros(len(time_steps)),
                 "EH": np.zeros(len(time_steps)),
                 }
        cooling_CC = np.zeros(len(time_steps))
        cooling_free = np.zeros(len(time_steps))
        res_heat_dem = np.zeros(len(time_steps))
        power_dem = np.zeros(len(time_steps))
        
        t_fc = np.zeros(len(time_steps))   # outlet temperature of free cooler



        for t in time_steps:
            # Calculate power demands of electrical heater and heat pump
            if nodes[n]["heat"][t] <= hp_capacity:
                power["HP"][t] = nodes[n]["heat"][t] / devs_dom["HP"]["COP"][n][t]
                power["EH"][t] = 0
            elif nodes[n]["heat"][t] > hp_capacity:
                power["HP"][t] = hp_capacity / devs_dom["HP"]["COP"][n][t]
                power["EH"][t] = (nodes[n]["heat"][t] - hp_capacity) / devs_dom["EH"]["eta_th"]
    
            # calculate load and power demand of compression chiller
            if param["T_hot"][t] + devs_dom["free_cooler"]["dT_min"] <= nodes[n]["T_cooling_return"][t]: # if network temperature is low enough, free cooling is possible
                t_fc[t] = max(nodes[n]["T_cooling_supply"][t], param["T_cold"][t] + devs_dom["free_cooler"]["dT_min"])
                cooling_free[t] = ( nodes[n]["T_cooling_return"][t] - t_fc[t] ) / (nodes[n]["T_cooling_return"][t] - nodes[n]["T_cooling_supply"][t]) * nodes[n]["cool"][t] 
                cooling_CC[t] = nodes[n]["cool"][t] - cooling_free[t]
                power["CC"][t] = cooling_CC[t] / devs_dom["CC"]["COP"][n][t]  
            else:
                cooling_CC[t] = nodes[n]["cool"][t]
                power["CC"][t] = cooling_CC[t] / devs_dom["CC"]["COP"][n][t]       
                
            res_heat_dem[t] = ((nodes[n]["heat"][t] - (power["EH"][t] * devs_dom["EH"]["eta_th"])) - 
                        power["HP"][t]) - (power["CC"][t] + nodes[n]["cool"][t]
                        # - air_cooling
                        )
            power_dem[t] = power["EH"][t] + power["CC"][t] + power["HP"][t]
       
        
        
        # Mass flow from hot to cold pipe
        mass_flow = res_heat_dem * 1000 / (param["c_f"] * (param["T_hot"] - param["T_cold"]))     # kg/s
        nodes[n]["mass_flow"] = mass_flow
        
        
        # save device capacities
        nodes[n]["HP_capacity"] = hp_capacity # kW_th
        nodes[n]["EH_capacity"] = eh_capacity # kW_th
        nodes[n]["CC_capacity"] = np.max(cooling_CC) # kW_th
        nodes[n]["free_cooler_capacity"] = np.max(cooling_free) #kW_th
        # nodes[n]["AirCooler_capacity"] =
        
        # Save node demands
        nodes[n]["res_heat_dem"] = res_heat_dem # kW_th
        nodes[n]["power_dem"] = power_dem
        
        # ANNUALIZED TBE COSTS
        devs = ["HP", "EH", "CC", "free_cooler"]
        nodes[n]["tac_building"] = sum(nodes[n][dev+"_capacity"] * devs_dom[dev]["inv_var"] * ( devs_dom[dev]["ann_factor"] + devs_dom[dev]["cost_om"] ) for dev in devs) / 1000
        
        

    param["tac_buildings"] = sum(nodes[n]["tac_building"] for n in nodes)
                

        

    return nodes, param