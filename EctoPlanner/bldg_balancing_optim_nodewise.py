# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:23:57 2018

@author: mwi
"""

import numpy as np
import gurobipy as gp
import time
import os
      
def run_optim(node, param, devs_dom, dir_results):
    
    
    t_air = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(0))          # Air temperature °C
    n = node["number"]
    time_steps = range(8760)
    
    # Initialize tac for building devices
    if n == 0:
        param["tac_buildings"] = 0
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    start_time = time.time()

    # Create set for devices
    all_devs = ["HP", "CC", "EH", "free_cooler", "air_cooler"]       
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("Building_Model_" + str(n))
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables
            
    # Device capacities
    cap = {}
    for device in all_devs:
        cap[device] = model.addVar(vtype = "C", name = "capacity_" + device)
    
    
    # Eletrical power to/from devices
    power = {}
    for device in ["HP", "CC", "EH"]:
        power[device] = {}
        for t in time_steps:
            power[device][t] = model.addVar(vtype="C", name="power_" + device + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["HP", "EH"]:
       heat[device] = {}
       for t in time_steps:
           heat[device][t] = model.addVar(vtype="C", name="heat_" + device + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "free_cooler", "air_cooler"]:
        cool[device] = {}
        for t in time_steps:
                cool[device][t] = model.addVar(vtype="C", name="cool_" + device + "_t" + str(t))
       
    
    # Mass flow in building cooling system
    m_cooling = {}
    m_free = {}
    m_air = {}
    m_rest = {}
    for t in time_steps:
        m_cooling[t] = model.addVar(vtype = "C", name="mass_flow_cooling_t" + str(t))
    for t in time_steps:
        m_free[t] = model.addVar(vtype = "C", name="mass_flow_free_cooler_t" + str(t))
    for t in time_steps:
        m_air[t] = model.addVar(vtype = "C", name="mass_flow_air_cooler_t" + str(t))
    for t in time_steps:
        m_rest[t] = model.addVar(vtype = "C", name="mass_flow_rest" + str(t))
    
                
    # Node residual loads
    res_el = {}
    for t in time_steps:
        res_el[t] = model.addVar(vtype="C", name="residual_power_demand_t" + str(t))
    res_thermal = {}
    for t in time_steps:
        res_thermal[t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_thermal_demand_t" + str(t))
    res_thermal_abs = {}
    for t in time_steps:
        res_thermal_abs[t] = model.addVar(vtype="C", name="residual_absolute_thermal_demand_t" + str(t))
    res_heat = {}
    for t in time_steps:
        res_heat[t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_heating_demand_t" + str(t)) 
    res_cool = {}
    for t in time_steps:
        res_cool[t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_cooling_demand_t" + str(t))   

    # Sum of residual loads
    sum_res_el = model.addVar(vtype = "C", name="sum_residual_power")
    sum_res_heat = model.addVar(vtype = "C", name="sum_residual_heating")   
    sum_res_cool = model.addVar(vtype = "C", name="sum_residual_cooling")  
    sum_res_thermal_abs =  model.addVar(vtype = "C", name="sum_residual_thermal_abs")   

    # annualized costs for building devices
    tac_building = model.addVar(vtype = "C", name = "tac_building")        
                
    # Objective functions
    obj = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="obj")    
        


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Define objective function
    model.update()
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    
 
    # Add constraints 
    
   
    #%% LOAD CONSTRAINTS
    
    for t in time_steps:
        for device in ["HP", "EH"]:
            model.addConstr(heat[device][t] <= cap[device])
        for device in ["CC", "air_cooler", "free_cooler"]:
            model.addConstr(cool[device][t] <= cap[device])
    
    
    #%% INPUT / OUTPUT CONSTRAINTS
    
    
    for t in time_steps:

        # Electric heater
        model.addConstr(heat["EH"][t] == power["EH"][t] * devs_dom["EH"]["eta_th"])
        
        # Compression chiller
        model.addConstr(cool["CC"][t] == power["CC"][t] * devs_dom["CC"]["COP"][n][t])  

        # Heat Pump
        model.addConstr(heat["HP"][t] == power["HP"][t] * devs_dom["HP"]["COP"][n][t])
                
        

    #%% ENERGY BALANCES
            
        
    for t in time_steps:
        
        # Heat balance
        model.addConstr(heat["EH"][t] + heat["HP"][t]  == node["heat"][t] )  

        # Cooling balance
        model.addConstr(cool["CC"][t] + cool["free_cooler"][t] + cool["air_cooler"][t] == node["cool"][t] ) 
        
        # Electricity balance
        model.addConstr(res_el[t] == power["EH"][t] + power["HP"][t] + power["CC"][t] )
            

    #%% FREE COOLING AND AIR COOLING RESTRICTIONS
    
    for t in time_steps:
        
        # Mass flow in cooling circle
        model.addConstr(m_cooling[t] == node["cool"][t] / (param["c_f"] * (node["T_cooling_return"][t] - node["T_cooling_supply"][t])) * 1000)
        
        # Sum of mass flows
        model.addConstr(m_cooling[t] == m_air[t] + m_free[t] + m_rest[t])
        
        # air cooling
        if t_air[t] + devs_dom["air_cooler"]["dT_min"] > node["T_cooling_return"][t]:
            model.addConstr(m_air[t] == 0)
            model.addConstr(cool["air_cooler"][t] == 0)
        else:
            model.addConstr(cool["air_cooler"][t] <= m_air[t] * param["c_f"]*(node["T_cooling_return"][t] - (t_air[t] + devs_dom["air_cooler"]["dT_min"])))
        
        # free cooling
        if param["T_hot"][t] + devs_dom["free_cooler"]["dT_min"] > node["T_cooling_return"][t]:
            model.addConstr(m_free[t] == 0)
            model.addConstr(cool["free_cooler"][t] == 0)
        else:
            model.addConstr(cool["free_cooler"][t] <= m_free[t] * param["c_f"]*(node["T_cooling_return"][t] - (param["T_cold"][t] + devs_dom["free_cooler"]["dT_min"])))
        
        
#        # air cooling
#        # if air temperature is too high, air cooling is not possible
#        if t_air[t] + devs_dom["air_cooler"]["dT_min"] >= node["T_cooling_return"][t]:
#            model.addConstr(t_ac[t] == node["T_cooling_return"][t])
#            model.addConstr(cool["air_cooler"][t] == 0)           
#        # else: cooling limit is t_air + dT_min
#        else:
#            model.addConstr(t_ac[t] >= t_air[t] + devs_dom["air_cooler"]["dT_min"])
#            model.addConstr(t_ac[t] >= node["T_cooling_supply"][t])
#            model.addConstr(cool["air_cooler"][t] == (node["T_cooling_return"][t] - t_ac[t])/(node["T_cooling_return"][t]-node["T_cooling_supply"][t]) * node["cool"][t])
#        
#        # free cooling
#        # if network temperature is too high, no free cooling is possible
#        if param["T_hot"][t] + devs_dom["free_cooler"]["dT_min"] >= node["T_cooling_return"][t]:
#            model.addConstr(t_fc[t] == node["T_cooling_return"][t])
#            model.addConstr(cool["free_cooler"][t] == 0)
#        else:
#            model.addConstr(t_fc[t] >= param["T_cold"][t] + devs_dom["free_cooler"]["dT_min"])
#            model.addConstr(t_fc[t] >= node["T_cooling_supply"][t])
#            model.addConstr(cool["free_cooler"][t] == (node["T_cooling_return"][t] - t_fc[t])/(node["T_cooling_return"][t]-node["T_cooling_supply"][t]) * node["cool"][t])
                                                  
            


    #%% RESIDUAL THERMAL LOADS
    
    for t in time_steps:
        
        model.addConstr(res_thermal[t] == (heat["HP"][t] - power["HP"][t]) - (cool["CC"][t] + power["CC"][t] + cool["free_cooler"][t] ))
        
        model.addConstr(res_thermal_abs[t] == gp.abs_(res_thermal[t]))
        
        model.addConstr(res_heat[t] == (res_thermal_abs[t] + res_thermal[t]) / 2 )
        model.addConstr(res_cool[t] == (res_thermal_abs[t] - res_thermal[t]) / 2 )
        


    #%% DEVICE RESTRICTIONS
    
    # electric heater
    if param["use_eh_in_bldgs"] == 0:  
        for t in time_steps:
            model.addConstr(heat["EH"] == 0 )    
            
    # either air-cooling or free-cooling
#    for t in time_steps:
#        model.addConstr(x_on["free_cooler"][t] + x_on["air_cooler"][t] <= 1)
#        model.addConstr(cool["free_cooler"][t] <= x_on["free_cooler"][t] * np.max(node["cool"]))
#        model.addConstr(cool["air_cooler"][t] <= x_on["air_cooler"][t] * np.max(node["cool"]))
            
            
    
    #%% SUM UP
    
    
    # Investment costs
    inv = {}
    for device in all_devs:
        inv[device] = devs_dom[device]["inv_var"]  * cap[device]
    
    # annualized investment
    c_inv = {}
    for device in all_devs:
        c_inv[device] = inv[device] * devs_dom[device]["ann_factor"]

    # Operation and maintenance costs
    c_om = {}
    for device in all_devs: 
        c_om[device] = devs_dom[device]["cost_om"] * inv[device]
    
    # Residual loads
    model.addConstr(sum_res_el == sum(res_el[t] for t in time_steps))
    model.addConstr(sum_res_heat == sum(res_heat[t] for t in time_steps))
    model.addConstr(sum_res_cool == sum(res_cool[t] for t in time_steps))
    model.addConstr(sum_res_thermal_abs == sum(res_thermal_abs[t] for t in time_steps))
    
    
    # Annualized device costs
    model.addConstr(tac_building ==  sum(c_inv[dev] for dev in all_devs) + sum(c_om[dev] for dev in all_devs))
    
    
            

    #%% OBJECTIVE

        
    model.addConstr( obj ==   tac_building
                                + sum_res_el * param["weight_el"]
                                + sum_res_heat * param["weight_heat"]
                                + sum_res_cool * param["weight_cool"]
                    )
            
#    model.addConstr( obj ==   sum_res_thermal_abs)

#%%
 # Set model parameters and execute calculation
    
    print("Precalculation and model set up done in %f seconds." %(time.time() - start_time))
    
    # Set solver parameters
    model.Params.MIPGap     = 0.01   # ---,         gap for branch-and-bound algorithm
    model.Params.method     = 2                 # ---,         -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc.
    model.Params.Heuristics = 0
    model.Params.MIPFocus   = 2
    model.Params.Cuts       = 3
    model.Params.PrePasses  = 8
    
    # Execute calculation
    start_time = time.time()

    model.optimize()

    print("Optimization done. (%f seconds.)" %(time.time() - start_time))
    
   
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Check and save results
    
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    # Check if optimal solution was found
    if model.Status in (3,4) or model.SolCount == 0:  # "INFEASIBLE" or "INF_OR_UNBD"
        model.computeIIS()
        model.write(dir_results + "\\" + "model.ilp")
        print('Optimization result: No feasible solution found.')
    
    else:
        
        # Write Gurobi files
        model.write(dir_results + "\model.lp")
        model.write(dir_results + "\model.prm")
        model.write(dir_results + "\model.sol")
        
        # get device capacities
#        cap = {}
#        for device in ["HP", "EH"]:
#            cap[device] = max(heat[device][t].X for t in time_steps)
#        for device in ["CC", "free_cooler", "air_cooler"]:
#            cap[device] = max(cool[device][t].X for t in time_steps)
#        
#        # Save device capacities in nodes 
        node["HP_capacity"] = cap["HP"].X
        node["CC_capacity"] = cap["CC"].X
        node["EH_capacity"] = cap["EH"].X
        node["air_cooler_capacity"] = cap["air_cooler"].X
        node["free_cooler_capacity"] = cap["free_cooler"].X
        
        # save residual loads in nodes
        node["res_heat_dem"] = np.zeros(8760)
        node["power_dem"] = np.zeros(8760)
        for t in time_steps:
            node["res_heat_dem"][t] = res_thermal[t].X
            node["power_dem"][t] = res_el[t].X       
        
        # Mass flow from hot to cold pipe
        mass_flow = node["res_heat_dem"] * 1000 / (param["c_f"] * (param["T_hot"] - param["T_cold"]))     # kg/s
        node["mass_flow"] = mass_flow
        
        
        # calculate annualized building device costs and add them to params
        # Investment costs
#        inv = {}
#        for device in all_devs:
#            inv[device] = devs_dom[device]["inv_var"] * cap[device]       
#        # annualized investment
#        c_inv = {}
#        for device in all_devs:
#            c_inv[device] = inv[device] * devs_dom[device]["ann_factor"]    
#        # Operation and maintenance costs
#        c_om = {}
#        for device in all_devs: 
#            c_om[device] = devs_dom[device]["cost_om"] * inv[device]
#            
#        tac_building = sum(c_inv[dev] for dev in all_devs) + sum(c_om[dev] for dev in all_devs)            
                
        node["tac_building"] = tac_building.X / 1000     # kEUR/a,   annualized costs for building devices
        
        param["tac_buildings"] += node["tac_building"]
        

    return node, param





          
                
    
    