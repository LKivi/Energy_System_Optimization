# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:23:57 2018

@author: mwi
"""

import numpy as np
import gurobipy as gp
import time
import os
      
def run_optim(nodes, param, devs_dom, dir_results):
    
    time_steps = range(8760)
    t_air = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(0))          # Air temperature °C
           
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    start_time = time.time()

    # Create set for devices
    all_devs = ["HP", "CC", "EH", "free_cooler", "air_cooler"]       
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("Balancing_Unit_Model")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["HP", "CC", "EH", "free_cooler", "air_cooler"]:
        cap[device] = {}
        for n in nodes:
            cap[device][n] = model.addVar(vtype="C", name="nominal_capacity_" + str(device) + "_n" + str(n))
            
     # Eletrical power to/from devices
    power = {}
    for device in ["HP", "CC", "EH"]:
        power[device] = {}
        for n in nodes:
            power[device][n] = {}
            for t in time_steps:
                power[device][n][t] = model.addVar(vtype="C", name="power_" + device + "_n" + str(n) + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["HP", "EH"]:
       heat[device] = {}
       for n in nodes:
           heat[device][n] = {}       
           for t in time_steps:
               heat[device][n][t] = model.addVar(vtype="C", name="heat_"  + device + "_n" + str(n) + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "free_cooler", "air_cooler"]:
        cool[device] = {}
        for n in nodes:
            cool[device][n] = {}
            for t in time_steps:
                cool[device][n][t] = model.addVar(vtype="C", name="cool_" + device + "_n" + str(n) + "_t" + str(t))
                
   
    # Mass flow in building cooling system
    m_cooling = {}
    m_free = {}
    m_air = {}
    m_rest = {}
    for n in nodes:
        m_cooling[n] = {}
        m_free[n] = {}
        m_air[n] = {}
        m_rest[n] = {}
        for t in time_steps:
            m_cooling[n][t] = model.addVar(vtype = "C", name="mass_flow_cooling_n" + str(n) + "_t" + str(t))
        for t in time_steps:
            m_free[n][t] = model.addVar(vtype = "C", name="mass_flow_free_cooler_n" + str(n) + "_t" + str(t))
        for t in time_steps:
            m_air[n][t] = model.addVar(vtype = "C", name="mass_flow_air_cooler_n" + str(n) + "_t" + str(t))
        for t in time_steps:
            m_rest[n][t] = model.addVar(vtype = "C", name="mass_flow_rest_n" + str(n) + "_t" + str(t))
                    
    # Node residual loads
    res_el = {}
    res_thermal= {}
    res_thermal_abs= {}
    res_heat= {}
    res_cool= {}
    for n in nodes:
        res_el[n] = {}
        for t in time_steps:
            res_el[n][t] = model.addVar(vtype="C", name="residual_power_demand_n" + str(n) + "_t" + str(t))
        res_thermal[n] = {}
        for t in time_steps:
            res_thermal[n][t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_thermal_demand_n" + str(n) + "_t" + str(t))
        res_thermal_abs[n] = {}
        for t in time_steps:
            res_thermal_abs[n][t] = model.addVar(vtype="C", name="residual_absolute_thermal_demand_n" + str(n) + "_t" + str(t))
        res_heat[n] = {}
        for t in time_steps:
            res_heat[n][t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_heating_demand_n" + str(n) + "_t" + str(t)) 
        res_cool[n] = {}
        for t in time_steps:
            res_cool[n][t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_cooling_demand_n" + str(n) + "_t" + str(t))   

    # Sum of residual loads
    sum_res_el = model.addVar(vtype = "C", name="sum_residual_power")
    sum_res_heat = model.addVar(vtype = "C", name="sum_residual_heating")   
    sum_res_cool = model.addVar(vtype = "C", name="sum_residual_cooling")  
    sum_res_thermal_abs = model.addVar(vtype = "C", name="sum_residual_thermal_abs")

    # Annualized technical building equipment costs
    tac_building = {}
    for n in nodes:
        tac_building[n] = model.addVar(vtype = "c", name="tac_building_n" + str(n))
              
                
    # Objective functions
    obj = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="obj")    
        


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Define objective function
    model.update()
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    
 
    # Add constraints 
    
    #%% LOAD CONSTRAINTS
    
    for n in nodes:  
        for t in time_steps:
            for device in ["HP", "EH"]:
                model.addConstr(heat[device][n][t] <= cap[device][n])
            
            for device in ["CC", "free_cooler", "air_cooler"]:
                model.addConstr(cool[device][n][t] <= cap[device][n])
    
    #%% INPUT / OUTPUT CONSTRAINTS
    
    for n in nodes:
        for t in time_steps:
    
            # Electric heater
            model.addConstr(heat["EH"][n][t] == power["EH"][n][t] * devs_dom["EH"]["eta_th"])
            
            # Compression chiller
            model.addConstr(cool["CC"][n][t] == power["CC"][n][t] * devs_dom["CC"]["COP"][n][t])  
    
            # Heat Pump
            model.addConstr(heat["HP"][n][t] == power["HP"][n][t] * devs_dom["HP"]["COP"][n][t])
                
        

    #%% ENERGY BALANCES
            
    for n in nodes:   
        for t in time_steps:
            
            # Heat balance
            model.addConstr(heat["EH"][n][t] + heat["HP"][n][t]  == nodes[n]["heat"][t] )  
    
            # Cooling balance
            model.addConstr(cool["CC"][n][t] + cool["free_cooler"][n][t] + cool["air_cooler"][n][t] == nodes[n]["cool"][t] ) 
            
            # Electricity balance
            model.addConstr(res_el[n][t] == power["EH"][n][t] + power["HP"][n][t] + power["CC"][n][t] )
            

    #%% FREE COOLING AND AIR COOLING RESTRICTIONS
    
    for n in nodes:
        for t in time_steps:
            
            # Mass flow in cooling circle
            model.addConstr(m_cooling[n][t] == nodes[n]["cool"][t] / (param["c_f"] * (nodes[n]["T_cooling_return"][t] - nodes[n]["T_cooling_supply"][t])) * 1000)
        
            # Sum of mass flows
            model.addConstr(m_cooling[n][t] == m_air[n][t] + m_free[n][t] + m_rest[n][t])
            
            # air cooling
            if t_air[t] + devs_dom["air_cooler"]["dT_min"] > nodes[n]["T_cooling_return"][t]:
                model.addConstr(m_air[n][t] == 0)
                model.addConstr(cool["air_cooler"][n][t] == 0)
            else:
                model.addConstr(cool["air_cooler"][n][t] <= m_air[n][t] * param["c_f"]*(nodes[n]["T_cooling_return"][t] - (t_air[t] + devs_dom["air_cooler"]["dT_min"])))
            
            # free cooling
            if param["T_hot"][t] + devs_dom["free_cooler"]["dT_min"] > nodes[n]["T_cooling_return"][t]:
                model.addConstr(m_free[n][t] == 0)
                model.addConstr(cool["free_cooler"][n][t] == 0)
            else:
                model.addConstr(cool["free_cooler"][n][t] <= m_free[n][t] * param["c_f"]*(nodes[n]["T_cooling_return"][t] - (param["T_cold"][t] + devs_dom["free_cooler"]["dT_min"])))
            
#            # air cooling
#            # if air temperature is too high, no air cooling is possible
#            if t_air[t] + devs_dom["air_cooler"]["dT_min"] >= nodes[n]["T_cooling_return"][t]:
#                model.addConstr(t_ac[n][t] == nodes[n]["T_cooling_return"][t])
#                model.addConstr(cool["air_cooler"][n][t] == 0)           
#            # else: cooling limit is t_air + dT_min
#            else:
#                model.addConstr(t_ac[n][t] >= t_air[t] + devs_dom["air_cooler"]["dT_min"])
#                model.addConstr(t_ac[n][t] >= nodes[n]["T_cooling_supply"][t])
#                model.addConstr(cool["air_cooler"][n][t] == (nodes[n]["T_cooling_return"][t] - t_ac[n][t])/(nodes[n]["T_cooling_return"][t]-nodes[n]["T_cooling_supply"][t]) * nodes[n]["cool"][t])
#            
#            # free cooling
#            # if network temperature is too high, no free cooling is possible
#            if param["T_hot"][t] + devs_dom["free_cooler"]["dT_min"] >= t_ac[n][t]:
#                model.addConstr(t_fc[n][t] == t_ac[n][t])
#                model.addConstr(cool["free_cooler"][n][t] == 0)
#            else:
#                model.addConstr(t_fc[n][t] >= param["T_cold"][t] + devs_dom["free_cooler"]["dT_min"])
#                model.addConstr(t_fc[n][t] >= nodes[n]["T_cooling_supply"][t])
#                model.addConstr(cool["free_cooler"][n][t] == (t_ac[n][t] - t_fc[n][t])/(nodes[n]["T_cooling_return"][t]-nodes[n]["T_cooling_supply"][t]) * nodes[n]["cool"][t])
        
            


    #%% RESIDUAL THERMAL LOADS
    
    for n in nodes:
        for t in time_steps:
            
            model.addConstr(res_thermal[n][t] == (heat["HP"][n][t] - power["HP"][n][t]) - (cool["CC"][n][t] + power["CC"][n][t] + cool["free_cooler"][n][t] ))
            
            model.addConstr(res_thermal_abs[n][t] == gp.abs_(res_thermal[n][t]))
            
            model.addConstr(res_heat[n][t] == (res_thermal_abs[n][t] + res_thermal[n][t]) / 2 )
            model.addConstr(res_cool[n][t] == (res_thermal_abs[n][t] - res_thermal[n][t]) / 2 )
        
    
    #%% DEVICE RESTRICTIONS
    
    if param["use_eh_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap["EH"][n] == 0 )
        
        
    
    #%% SUM UP

    # Investment costs
    inv = {}
    c_inv = {}
    c_om = {}
 
    for device in all_devs:
        inv[device] = {}
        c_inv[device] = {}
        c_om[device] = {}

        for n in nodes:
            
            # investment costs
            inv[device][n] = devs_dom[device]["inv_var"] * cap[device][n]
        
            # annualized investment
            c_inv[device][n] = inv[device][n] * devs_dom[device]["ann_factor"]
    
            # Operation and maintenance costs 
            c_om[device][n] = devs_dom[device]["cost_om"] * inv[device][n]
            
    
    # annualized costs for building devices
    for n in nodes:
        model.addConstr(tac_building[n] ==  sum(c_inv[dev][n] for dev in all_devs) + sum(c_om[dev][n] for dev in all_devs))    
    
    # Residual loads
    for t in time_steps:
        model.addConstr(sum_res_el[t] == sum(sum(res_el[n][t] for n in nodes) for t in time_steps))
    for t in time_steps:    
        model.addConstr(sum_res_heat[t] == sum(sum(res_heat[n][t] for n in nodes) for t in time_steps))
    for t in time_steps:   
        model.addConstr(sum_res_cool[t] == sum(sum(res_cool[n][t] for n in nodes)for t in time_steps))
    for t in time_steps:   
        model.addConstr(sum_res_thermal_abs[t] == sum(sum(res_cool[n][t] for n in nodes) for t in time_steps))
    
    
    
            

    #%% OBJECTIVE
    
    
    model.addConstr(obj ==       sum(tac_building[n] for n in nodes)   
                                    + sum_res_el * param["weight_el"]
                                    + sum_res_heat * param["weight_heat"]
                                    + sum_res_cool * param["weight_cool"]
                                    )
        
            
        

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
        
        # Save device capacities in nodes 
        for n in nodes:
            nodes[n]["HP_capacity"] = cap["HP"][n].X
            nodes[n]["CC_capacity"] = cap["CC"][n].X
            nodes[n]["EH_capacity"] = cap["EH"][n].X
            nodes[n]["air_cooler_capacity"] = cap["air_cooler"][n].X
            nodes[n]["free_cooler_capacity"] = cap["free_cooler"][n].X
            
            # save residual loads in nodes
            nodes[n]["res_heat_dem"] = np.zeros(8760)
            nodes[n]["power_dem"] = np.zeros(8760)
            for t in time_steps:
                nodes[n]["res_heat_dem"][t] = res_thermal[n][t].X
                nodes[n]["power_dem"][t] = res_el[n][t].X       
        
            # Mass flow from hot to cold pipe
            mass_flow = nodes[n]["res_heat_dem"] * 1000 / (param["c_f"] * (param["T_hot"] - param["T_cold"]))     # kg/s
            nodes[n]["mass_flow"] = mass_flow
            
        
            nodes[n]["tac_building"] = tac_building[n].X / 1000
        
        
        # save annualized costs for devices
        param["tac_buildings"] = sum(nodes[n]["tac_building"] for n in nodes)          # kEUR/a, annualized costs for building devices
        
        

    return nodes, param





          
                
    
    