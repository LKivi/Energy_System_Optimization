# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20

@author: lkivi
"""

import numpy as np
import gurobipy as gp
import time
import os
import post_processing_clustered as post
      

def run(nodes, param, devs, devs_dom, dir_results):
    
    if param["feasible_TES"] and param["switch_combined_storage"]:
        
        devs = run_optim(nodes, param, devs, devs_dom, dir_results, 1)
        
        nodes, param = run_optim(nodes, param, devs, devs_dom, dir_results, 2)
        
    else:
        
        nodes, param = run_optim(nodes, param, devs, devs_dom, dir_results, 1)
        
    return nodes, param
        
        
        
        

#%%
# consider integrated hot and cold storage
def run_optim(nodes, param, devs, devs_dom, dir_results, step):
    
    days = range(param["n_clusters"])
    time_steps = range(24)
    
    t_air = param["t_air"]         # Air temperature °C           
    G_sol = param["G_sol"]        # Solar radiation W/m^2              
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    start_time = time.time()

    # Create set for BU devices
    all_devs = ["BOI", "CHP", "AC", "CC", "TES", "air_cooler", "HP", "EH"] 
    
    if step == 1:
        all_devs.append("CTES")
    
    # Create set for building devices
    all_devs_dom = ["HP", "CC", "EH", "free_cooler", "air_cooler", "BOI", "PV", "TES"]     
    
    
    ## Simplification of BU-Optimization
    
    # Get constant investment costs (kEUR / MW)
    inv_var = {}
    inv_var["BOI"] = 67.5
    inv_var["CHP"] = 541.5
    inv_var["EH"] = 150
    inv_var["AC"] = 360
    inv_var["CC"] = 150
    inv_var["HP"] = 120
    inv_var["TES"] = 9.2             # kEUR/MWh
    inv_var["CTES"] = 55.2            # kEUR/MWh
    inv_var["air_cooler"] = 37.5
    
    
    # Get time series of storage function
    if step == 2:
        x = {}
        x["TES"] = {}
        x["CTES"] = {}
        for t in range(8761):
            x["TES"][t] = devs["TES"]["x"][t]
            x["CTES"][t] = devs["CTES"]["x"][t]
        
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("Global_Optimization")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables
    
    #%% BALANCING UNIT VARIABLES
    
    # Piece-wise linear function variables
    lin = {}
    for device in ["BOI", "CHP", "AC", "CC", "EH", "TES", "CTES", "air_cooler", "HP"]:   
        lin[device] = {}
        for i in range(len(devs[device]["cap_i"])):
            lin[device][i] = model.addVar(vtype="C", name="lin_" + device + "_i" + str(i))
    
    # Purchase decision binary variables (1 if device is installed, 0 otherwise)
#    x = {}
#    for device in all_devs:
#        x[device] = model.addVar(vtype="B", name="x_" + str(device))
            
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "air_cooler", "HP", "EH"]:
        cap[device] = model.addVar(vtype="C", name="nominal_capacity_" + str(device))
        
    # Storage volumes
    vol = {}
    for device in ["TES", "CTES"]:
        vol[device] = model.addVar(vtype = "C", name="volume_" + str(device))
    
    
    # Gas flow to/from devices
    gas = {}
    for device in ["BOI", "CHP", "to_buildings"]:
        gas[device] = {}
        for d in days:
            gas[device][d] = {}
            for t in time_steps:
                gas[device][d][t] = model.addVar(vtype="C", name="gas_" + device + "_d" + str(d) + "_t" + str(t))
        
    # Eletrical power to/from devices
    power = {}
    for device in ["CHP", "CC", "from_grid", "to_grid", "HP", "EH", "PV"]:
        power[device] = {}
        for d in days:
            power[device][d] = {}
            for t in time_steps:
                power[device][d][t] = model.addVar(vtype="C", name="power_" + device + "_d" + str(d) + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["BOI", "CHP", "AC", "HP", "EH"]:
        heat[device] = {}
        for d in days:
            heat[device][d] = {}
            for t in time_steps:
                heat[device][d][t] = model.addVar(vtype="C", name="heat_" + device + "_d" + str(d) + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "AC", "air_cooler"]:
        cool[device] = {}
        for d in days:
            cool[device][d] = {}
            for t in time_steps:
                cool[device][d][t] = model.addVar(vtype="C", name="cool_" + device + "_d" + str(d) + "_t" + str(t))
            
    # grid maximum transmission power
    grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
    grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")
    
    # total energy amounts taken from grid and fed into grid
    from_grid_total = model.addVar(vtype = "C", name="from_grid_total")
    # total power to grid
    to_grid_total = model.addVar(vtype = "C", name="to_grid_total")
    gas_total = model.addVar(vtype = "C", name="gas_total")
    # total revenue for feed-in
    electricity_costs_total = model.addVar(vtype = "C", name="electricity_costs_total")
    

    # Storage variables
    
    # initial state of charge which has to be followed at the beginning (and end) of every type-day
    soc_init = {}
    for device in ["TES", "CTES"]:
        soc_init[device] = model.addVar(vtype="C", name="initial_soc_" + device)
      
    ch = {}  # Energy flow to charge storage device
    dch = {} # Energy flow to discharge storage device
    soc = {} # State of charge
#    x = {}   # decision varaible determining the function of the storage (hot/cold)

    for device in ["TES", "CTES"]:
        ch[device] = {}
        dch[device] = {}
        soc[device] = {}
#        x[device] = {}
        for d in days:
            ch[device][d] = {}
            for t in time_steps:
                ch[device][d][t] = model.addVar(vtype="C", name="ch_" + device + "_d" + str(d) + "_t" + str(t))
        for d in days:
            dch[device][d] = {}
            for t in time_steps:
                dch[device][d][t] = model.addVar(vtype="C", name="dch_" + device + "_d" + str(d) + "_t" + str(t))
        for d in days:
            soc[device][d] = {}
            for t in time_steps:
                soc[device][d][t] = model.addVar(vtype="C", name="soc_" + device + "_d" + str(d) + "_t" + str(t))
            soc[device][d][len(time_steps)] = model.addVar(vtype="C", name="soc_" + device + "_d" + str(d) + "_t" + str(len(time_steps)))
              
#        for i in intervals:
#            x[device][i] = model.addVar(vtype="B", name="storage_is_" + device + "_i" + str(i))
        
    # Investment costs
    inv = {}
    c_inv = {}
    c_om = {}    
    for device in all_devs:
        inv[device] = model.addVar(vtype = "C", name = "inv_costs_" + device)
        c_inv[device] = model.addVar(vtype = "C", name = "annual_inv_costs" + device)
        c_om[device] = model.addVar(vtype = "C", name = "om_costs" + device)         
        
    
    
    
    #%% BUILDING VARIABLES
    
    # Device's capacity (i.e. nominal power)
    cap_dom = {}
    for device in ["HP", "CC", "EH", "free_cooler", "air_cooler", "BOI", "PV", "TES"]:
        cap_dom[device] = {}
        for n in nodes:
            cap_dom[device][n] = model.addVar(vtype="C", name="nominal_capacity_" + str(device) + "_n" + str(n))
            
    # PV roof area
    area_dom = {}
    for device in ["PV"]:
        area_dom[device] = {}
        for n in nodes:
            area_dom[device][n] = model.addVar(vtype = "C", name="roof_area_" + str(device) + "_n" + str(n))
            
    # TES volumes
    vol_dom = {}
    for device in ["TES"]:
        vol_dom[device] = {}
        for n in nodes:
            vol_dom[device][n] = model.addVar(vtype = "C", name= "volume_" + str(device) + "_n" + str(n))
    
    # Eletrical power to/from devices
    power_dom = {}
    for device in ["HP", "CC", "EH", "PV"]:
        power_dom[device] = {}
        for n in nodes:
            power_dom[device][n] = {}
            for d in days:
                power_dom[device][n][d] = {}
                for t in time_steps:
                    power_dom[device][n][d][t] = model.addVar(vtype="C", name="power_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))

     # Gas to devices
    gas_dom = {}
    for device in ["BOI"]:
        gas_dom[device] = {}
        for n in nodes:
            gas_dom[device][n] = {}
            for d in days:
                gas_dom[device][n][d] = {}
                for t in time_steps:
                    gas_dom[device][n][d][t] = model.addVar(vtype="C", name="gas_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))
       
    # Heat to/from devices
    heat_dom = {}
    for device in ["HP", "EH", "BOI"]:
       heat_dom[device] = {}
       for n in nodes:
           heat_dom[device][n] = {}  
           for d in days:
               heat_dom[device][n][d] = {}
               for t in time_steps:
                   heat_dom[device][n][d][t] = model.addVar(vtype="C", name="heat_"  + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))
    
    # Cooling power to/from devices
    cool_dom = {}
    for device in ["CC", "free_cooler", "air_cooler"]:
        cool_dom[device] = {}
        for n in nodes:
            cool_dom[device][n] = {}
            for d in days:
                cool_dom[device][n][d] = {}
                for t in time_steps:
                    cool_dom[device][n][d][t] = model.addVar(vtype="C", name="cool_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))
                
    # Storage variables 

    # initial soc of every type-day
    soc_dom_init = {}
    for device in ["TES"]: 
        soc_dom_init[device] = {}
        for n in nodes:
            soc_dom_init[device][n] = model.addVar(vtype="C", name="initial_soc_" + device + "_n" + str(n))
    
    ch_dom = {}                 # Energy flow to charge storage device
    dch_dom = {}                # Energy flow to discharge storage device
    soc_dom = {}                # State of charge
    for device in ["TES"]:
        ch_dom[device] = {}
        dch_dom[device] = {}
        soc_dom[device] = {}
        for n in nodes:
            ch_dom[device][n] = {}
            dch_dom[device][n] = {}
            soc_dom[device][n] = {} 
            for d in days:
                ch_dom[device][n][d] = {}
                for t in time_steps:
                    ch_dom[device][n][d][t] = model.addVar(vtype="C", name="ch_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))
            for d in days:
                dch_dom[device][n][d] = {}
                for t in time_steps:
                    dch_dom[device][n][d][t] = model.addVar(vtype="C", name="dch_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))
            for d in days:
                soc_dom[device][n][d] = {} 
                for t in time_steps:
                    soc_dom[device][n][d][t] = model.addVar(vtype="C", name="soc_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))
                soc_dom[device][n][d][len(time_steps)] = model.addVar(vtype="C", name="soc_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(len(time_steps)))
                
   
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
        for d in days:
            m_cooling[n][d] = {}
            for t in time_steps:
                m_cooling[n][d][t] = model.addVar(vtype = "C", name="mass_flow_cooling_n" + str(n) + "_d" + str(d) + "_t" + str(t))
        for d in days:
            m_free[n][d] = {}
            for t in time_steps:
                m_free[n][d][t] = model.addVar(vtype = "C", name="mass_flow_free_cooler_n" + str(n) + "_d" + str(d) + "_t" + str(t))
        for d in days:
            m_air[n][d] = {}
            for t in time_steps:
                m_air[n][d][t] = model.addVar(vtype = "C", name="mass_flow_air_cooler_n" + str(n) + "_d" + str(d) + "_t" + str(t))
        for d in days:
            m_rest[n][d] = {}
            for t in time_steps:
                m_rest[n][d][t] = model.addVar(vtype = "C", name="mass_flow_rest_n" + str(n) + "_d" + str(d) + "_t" + str(t))

                    
    
    # Node residual loads
    res_el = {}
    res_thermal= {}
#    res_thermal_abs= {}
#    res_heat= {}
#    res_cool= {}
    for n in nodes:
        res_el[n] = {}
        for d in days:
            res_el[n][d] = {}
            for t in time_steps:
                res_el[n][d][t] = model.addVar(vtype="C", name="residual_power_supply_n" + str(n) + "_d" + str(d) + "_t" + str(t))
        res_thermal[n] = {}
        for d in days:
            res_thermal[n][d] = {}
            for t in time_steps:
                res_thermal[n][d][t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_thermal_demand_n" + str(n) + "_d" + str(d) + "_t" + str(t))
#        res_thermal_abs[n] = {}
#        for d in days:
#            res_thermal_abs[n][d] = {}
#            for t in time_steps:
#                res_thermal_abs[n][d][t] = model.addVar(vtype="C", name="residual_absolute_thermal_demand_n" + str(n) + "_d" + str(d) + "_t" + str(t))
#        res_heat[n] = {}
#        for d in days:
#            res_heat[n][d] = {}
#            for t in time_steps:
#                res_heat[n][d][t] = model.addVar(vtype="C", name="residual_heating_demand_n" + str(n) + "_d" + str(d) + "_t" + str(t))             
#        res_cool[n] = {}
#        for d in days:
#            res_cool[n][d] = {}
#            for t in time_steps:
#                res_cool[n][d][t] = model.addVar(vtype="C", name="residual_cooling_demand_n" + str(n) + "_d" + str(d) + "_t" + str(t))   

    
    # Total residual network load
    residual = {}  
    residual["power"] = {}
    residual["heat"] = {}
    residual["cool"] = {}
    residual["thermal"] = {}
    residual["thermal_abs"] = {}
    for d in days:
        residual["power"][d] = {}
        for t in time_steps:        
            residual["power"][d][t] = model.addVar(vtype = "C", name="residual_power_d" + str(d) + "_t" + str(t))
    for d in days:
        residual["heat"][d] = {}
        for t in time_steps:
            residual["heat"][d][t] = model.addVar(vtype = "C", name="residual_heating_d" + str(d) + "_t" + str(t))   
    for d in days:
        residual["cool"][d] = {}
        for t in time_steps:
            residual["cool"][d][t] = model.addVar(vtype = "C", name="residual_cooling_d" + str(d) + "_t" + str(t))
    for d in days:
        residual["thermal"][d] = {}
        for t in time_steps:
            residual["thermal"][d][t] = model.addVar(lb = -gp.GRB.INFINITY, vtype = "C", name="residual_thermal_d" + str(d) + "_t" + str(t))        
    for d in days:
        residual["thermal_abs"][d] = {}
        for t in time_steps:
            residual["thermal_abs"][d][t] = model.addVar(vtype = "C", name="residual_thermal_abs_d" + str(d) + "_t" + str(t))
        
        
    # Investment costs
    inv_dom = {}
    c_inv_dom = {}
    c_om_dom = {}
    for device in all_devs_dom:
        inv_dom[device] = {}
        c_inv_dom[device] = {}
        c_om_dom[device] = {} 
        for n in nodes:
            inv_dom[device][n] = model.addVar(vtype="C", name="inv_costs_" + device + "_n" + str(n))
        for n in nodes:
            c_inv_dom[device][n] = model.addVar(vtype="C", name="annual_inv_costs_" + device + "_n" + str(n))
        for n in nodes:
            c_om_dom[device][n]  = model.addVar(vtype="C", name="om_costs_" + device + "_n" + str(n))             

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
    
    
    #%% BUILDING CONSTRAINTS
    
    #%% Capacity constraints
    
    # PV
    for n in nodes:
        for device in ["PV"]:
            model.addConstr(cap_dom[device][n] <= devs_dom[device]["max_cap"][n])
            model.addConstr(area_dom[device][n] <= devs_dom[device]["max_area"][n])
            
    # Thermal storage
    for n in nodes:
        for device in ["TES"]:
            model.addConstr(cap_dom[device][n] <= devs_dom[device]["max_cap"][n])
            model.addConstr(cap_dom[device][n] >= devs_dom[device]["min_cap"][n])
            # Relation between capacity and volume
            model.addConstr(cap_dom[device][n] == vol_dom[device][n] * param["rho_f"] *  param["c_f"] * (devs_dom[device]["T_max"] - devs_dom[device]["T_min"]) / (1000 * 3600))
            
    
    #%% LOAD CONSTRAINTS
    
    for n in nodes:
        for device in ["TES"]:
            for d in days:
                for t in time_steps:    
                        model.addConstr(soc_dom[device][n][d][t] <= cap_dom[device][n])
                        model.addConstr(ch_dom[device][n][d][t] <= nodes[n]["heat"][d][t])
                        model.addConstr(dch_dom[device][n][d][t] <= nodes[n]["heat"][d][t])
        model.addConstr(soc_dom_init[device][n] <= cap_dom[device][n])  
    
    for n in nodes:
        for d in days:
            for t in time_steps:
                for device in ["PV"]:
                    model.addConstr(power_dom[device][n][d][t] <= cap_dom[device][n])                
                
                for device in ["HP", "EH", "BOI"]:
                    model.addConstr(heat_dom[device][n][d][t] <= cap_dom[device][n])
                
                for device in ["CC", "free_cooler", "air_cooler"]:
                    model.addConstr(cool_dom[device][n][d][t] <= cap_dom[device][n])
                
            # heat pump is limited by maximum supply temperature
#            for device in ["HP"]:
#                model.addConstr(heat_dom[device][n][t] <= nodes[n]["heat"][t] * (devs_dom["HP"]["T_supply_max"] - nodes[n]["T_heating_return"][t])/(nodes[n]["T_heating_supply"][t] - nodes[n]["T_heating_return"][t]))
    
    #%% INPUT / OUTPUT CONSTRAINTS
    
    for n in nodes:
        for d in days:
            for t in time_steps:
        
                # Electric heater
                model.addConstr(heat_dom["EH"][n][d][t] == power_dom["EH"][n][d][t] * devs_dom["EH"]["eta_th"])
                
                # Compression chiller
                model.addConstr(cool_dom["CC"][n][d][t] == power_dom["CC"][n][d][t] * devs_dom["CC"]["COP"][n][d][t])  
        
                # Heat Pump
                model.addConstr(heat_dom["HP"][n][d][t] == power_dom["HP"][n][d][t] * devs_dom["HP"]["COP"][n][d][t])
                
                # Boiler
                model.addConstr(heat_dom["BOI"][n][d][t] == gas_dom["BOI"][n][d][t] * devs_dom["BOI"]["eta_th"])
                
                # PV
                model.addConstr(power_dom["PV"][n][d][t] <= G_sol[d][t]/1000 * devs_dom["PV"]["eta_el"] * area_dom["PV"][n])            
                
        

    #%% ENERGY BALANCES
            
    for n in nodes:
        for d in days:
            for t in time_steps:
                
                # Heat balance
                model.addConstr(heat_dom["EH"][n][d][t] + heat_dom["HP"][n][d][t] + heat_dom["BOI"][n][d][t] + dch_dom["TES"][n][d][t]  == nodes[n]["heat"][d][t] + ch_dom["TES"][n][d][t] )  
        
                # Cooling balance
                model.addConstr(cool_dom["CC"][n][d][t] + cool_dom["free_cooler"][n][d][t] + cool_dom["air_cooler"][n][d][t] == nodes[n]["cool"][d][t] ) 
                
                # Electricity demands
                model.addConstr(res_el[n][d][t] == power_dom["EH"][n][d][t] + power_dom["HP"][n][d][t] + power_dom["CC"][n][d][t])
                                
                # Thermal storage can only be supplied by Electric heater and boiler
                model.addConstr(heat_dom["EH"][n][d][t] + heat_dom["BOI"][n][d][t] >= ch_dom["TES"][n][d][t])
            
    
    #%% BUILDING THERMAL STORAGES

    for n in nodes:
    
        for device in ["TES"]:
            
            for d in days:
                            
                # Initial state of charge
                model.addConstr(soc_dom[device][n][d][0] <= soc_dom_init[device][n])
                
                # Cyclic condition
                model.addConstr(soc_dom[device][n][d][len(time_steps)] == soc_dom[device][n][d][0])
            
                for t in np.arange(1,len(time_steps)+1):
                    # Energy balance: soc(t) = soc(t-1) + charge - discharge
                    model.addConstr(soc_dom[device][n][d][t] == soc_dom[device][n][d][t-1] * (1-devs_dom[device]["sto_loss"])
                        + (ch_dom[device][n][d][t-1] * devs_dom[device]["eta_ch"] 
                        - dch_dom[device][n][d][t-1] / devs_dom[device]["eta_dch"]))
            

    #%% FREE COOLING AND AIR COOLING RESTRICTIONS
    
    for n in nodes:
        
        for d in days:
            
            for t in time_steps:
                
                # Mass flow in cooling circle
                model.addConstr(m_cooling[n][d][t] == nodes[n]["cool"][d][t] / (param["c_f"] * (nodes[n]["T_cooling_return"][d][t] - nodes[n]["T_cooling_supply"][d][t])) * 1000)
            
                # Sum of mass flows
                model.addConstr(m_cooling[n][d][t] == m_air[n][d][t] + m_free[n][d][t] + m_rest[n][d][t])
                
                # air cooling
                if t_air[d][t] + devs_dom["air_cooler"]["dT_min"] > nodes[n]["T_cooling_return"][d][t]:
                    model.addConstr(m_air[n][d][t] == 0)
                    model.addConstr(cool_dom["air_cooler"][n][d][t] == 0)
                else:
                    model.addConstr(cool_dom["air_cooler"][n][d][t] <= m_air[n][d][t] * param["c_f"] * (nodes[n]["T_cooling_return"][d][t] - (t_air[d][t] + devs_dom["air_cooler"]["dT_min"])) / 1000 ) 
                
                # free cooling
                if param["T_hot"][d][t] + devs_dom["free_cooler"]["dT_min"] > nodes[n]["T_cooling_return"][d][t]:
                    model.addConstr(m_free[n][d][t] == 0)
                    model.addConstr(cool_dom["free_cooler"][n][d][t] == 0)
                else:
                    model.addConstr(cool_dom["free_cooler"][n][d][t] <= m_free[n][d][t] * param["c_f"] * (nodes[n]["T_cooling_return"][d][t] - (param["T_cold"][d][t] + devs_dom["free_cooler"]["dT_min"])) / 1000 )
 

           

    #%% RESIDUAL THERMAL LOADS
    
    for n in nodes:
        for d in days:
            
            for t in time_steps:                
                model.addConstr(res_thermal[n][d][t] == (heat_dom["HP"][n][d][t] - power_dom["HP"][n][d][t]) - (cool_dom["CC"][n][d][t] + power_dom["CC"][n][d][t] + cool_dom["free_cooler"][n][d][t] ))
        
    
    #%% DEVICE RESTRICTIONS

    if param["use_cc_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["CC"][n] == 0 )
    
    if param["use_eh_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["EH"][n] == 0 )
            
    if param["use_pv_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["PV"][n] == 0 )            
            
    if param["use_boi_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["BOI"][n] == 0 )
            
    if param["use_free_cooler_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["free_cooler"][n] == 0 )

    if param["use_air_cooler_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["air_cooler"][n] == 0 )
            
    if param["use_tes_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["TES"][n] == 0 )
            for d in days:
                for t in time_steps:
                    model.addConstr( ch_dom["TES"][n][d][t] == 0)
                    model.addConstr( dch_dom["TES"][n][d][t] == 0)
                    model.addConstr( soc_dom["TES"][n][d][t] == 0)
            
            
            
            
    #%% SUM UP

    # Investment costs
 
    for device in all_devs_dom:

        for n in nodes:
            
            # investment costs
            model.addConstr( inv_dom[device][n] == devs_dom[device]["inv_var"] * cap_dom[device][n] + devs_dom[device]["inv_fix"] )
        
            # annualized investment
            model.addConstr( c_inv_dom[device][n] == devs_dom[device]["ann_inv_var"] * cap_dom[device][n] + devs_dom[device]["ann_inv_fix"] )
    
            # Operation and maintenance costs 
            model.addConstr( c_om_dom[device][n] == devs_dom[device]["cost_om"] * inv_dom[device][n] )
            
    
    # annualized costs for building devices
    for n in nodes:
        model.addConstr(tac_building[n] ==  (sum(c_inv_dom[dev][n] for dev in all_devs_dom) + sum(c_om_dom[dev][n] for dev in all_devs_dom)) / 1000)    
    
    # Residual loads
    for d in days:    
        for t in time_steps:
            model.addConstr(residual["power"][d][t] == sum(res_el[n][d][t] for n in nodes) / 1000)
    for d in days:    
        for t in time_steps:
            model.addConstr(residual["thermal"][d][t] == sum(res_thermal[n][d][t] for n in nodes) / 1000)

    # if two heating and cooling balances are considered: seperate residual thermal load into heating and cooling load
    if not param["switch_single_balance"]:        
        for d in days:
            for t in time_steps: 
                                              
                model.addGenConstrAbs(residual["thermal_abs"][d][t], residual["thermal"][d][t])
                
                model.addConstr(residual["heat"][d][t] == (residual["thermal_abs"][d][t] + residual["thermal"][d][t]) / 2 )
                model.addConstr(residual["cool"][d][t] == (residual["thermal_abs"][d][t] - residual["thermal"][d][t]) / 2 )    
                
    # Gas supply for building boilers
    for d in days:
        for t in time_steps:
            model.addConstr(gas["to_buildings"][d][t] == sum(gas_dom["BOI"][n][d][t] for n in nodes) / 1000)
            
    # PV generation in buildings
    for d in days:
        for t in time_steps:
            model.addConstr(power["PV"][d][t] == sum(power_dom["PV"][n][d][t] for n in nodes) / 1000)

        
        
        
    #%% BALANCING UNIT CONSTRAINTS
        
   
    #%% CAPACITY CONSTRAINTS
    
    for device in ["TES", "CTES"]:
        model.addConstr(cap[device] <= devs[device]["max_cap"])
        model.addConstr(cap[device] >= devs[device]["min_cap"])
        # Relation between volume and capacity
        model.addConstr(cap[device] == vol[device] * param["rho_f"] *  param["c_f"] * (devs[device]["T_max"] - devs[device]["T_min"]) / (1e6 * 3600))
               
    if step == 2:
        model.addConstr(vol["TES"] == vol["CTES"])

    #%% LOAD CONSTRAINTS
    
    for device in ["TES", "CTES"]:
        for d in days:
            for t in time_steps:    
                model.addConstr(soc[device][d][t] <= cap[device])
        model.addConstr(soc_init[device] <= cap[device])
        
    for t in time_steps:
        for d in days:
            for device in ["BOI", "HP", "EH"]:
                model.addConstr(heat[device][d][t] <= cap[device])
                
            for device in ["CHP"]:
                model.addConstr(power[device][d][t] <= cap[device])
            
            for device in ["CC", "AC", "air_cooler"]:
                model.addConstr(cool[device][d][t] <= cap[device])
    
            # limitation of power from and to grid   
            model.addConstr(sum(gas[device][d][t] for device in ["BOI", "CHP", "to_buildings"]) <= grid_limit_gas)       
            for device in ["from_grid", "to_grid"]:
                model.addConstr(power[device][d][t] <= grid_limit_el)
            
    # Air cooler temperature constraints
    for d in days:
        for t in time_steps:
            if t_air[d][t] + devs_dom["air_cooler"]["dT_min"] > param["T_hot"][d][t]:
                model.addConstr(cool["air_cooler"][d][t] == 0)
            else:
                if param["switch_single_balance"]:
                    model.addConstr( cool["air_cooler"][d][t] <= ( -residual["thermal"][d][t] + heat["BOI"][d][t] + heat["CHP"][d][t] + heat["HP"][d][t] + heat["EH"][d][t] + dch["TES"][d][t]) * (param["T_hot"][d][t] - (t_air[d][t] + devs_dom["air_cooler"]["dT_min"]))/ (param["T_hot"][d][t] - param["T_cold"][d][t] ))
                else:
                    model.addConstr( cool["air_cooler"][d][t] <=  residual["cool"][d][t] * (param["T_hot"][d][t] - (t_air[d][t] + devs_dom["air_cooler"]["dT_min"]))/ (param["T_hot"][d][t] - param["T_cold"][d][t] ))
            
    #%% INPUT / OUTPUT CONSTRAINTS
    for t in time_steps:
        
        for d in days:
            
            # Boiler
            model.addConstr(gas["BOI"][d][t] == heat["BOI"][d][t] / devs["BOI"]["eta_th"])
            
            # Heat pump
            model.addConstr(heat["HP"][d][t] == power["HP"][d][t] * devs["HP"]["COP"][d][t])
            
            # Combined heat and power
            model.addConstr(power["CHP"][d][t] == heat["CHP"][d][t] / devs["CHP"]["eta_th"] * devs["CHP"]["eta_el"])
            model.addConstr(gas["CHP"][d][t] == heat["CHP"][d][t] / devs["CHP"]["eta_th"])
            
            # Electric heater
            model.addConstr(heat["EH"][d][t] == power["EH"][d][t] * devs["EH"]["eta_th"])
            
            # Compression chiller
            model.addConstr(cool["CC"][d][t] == power["CC"][d][t] * devs["CC"]["COP"][d][t])  
    
            # Absorption chiller
            model.addConstr(cool["AC"][d][t] == heat["AC"][d][t] * devs["AC"]["eta_th"])
        
    
     #%% STORAGE DEVICES
    
    
    for device in ["TES", "CTES"]:
        
        for d in days:
        
            # Initial state of charge
            model.addConstr(soc[device][d][0] == soc_init[device])           
            
            # Cyclic condition
            model.addConstr(soc[device][d][len(time_steps)] == soc[device][d][0])
        
        
            if step == 1:
        
                for t in np.arange(1,len(time_steps)+1):
                    # Energy balance: soc(t) = soc(t-1) + charge - discharge
                    model.addConstr(soc[device][d][t] == soc[device][d][t-1] * (1-devs[device]["sto_loss"])
                        + (ch[device][d][t-1] * devs[device]["eta_ch"] 
                        - dch[device][d][t-1] / devs[device]["eta_dch"]))
                    
    
       
        
        if step == 2:
           
            for t in range(len(time_steps)+1):        
                
                if t == 0:
                    # Set initial state of charge
                    model.addConstr(soc[device][0] <= cap[device] * devs[device]["soc_init"])
                    model.addConstr(soc[device][0] <= x[device][0] * devs[device]["max_cap"])
                    model.addConstr(soc[device][0] >= x[device][0] * devs[device]["min_cap"])
                    
                else:
                    # Energy balance: soc(t) = soc(t-1) + charge - discharge
                    model.addConstr(soc[device][t] == soc[device][t-1] * (1-devs[device]["sto_loss"])
                        + (ch[device][t-1] * devs[device]["eta_ch"] 
                        - dch[device][t-1] / devs[device]["eta_dch"]))
                    
                    
                    # soc_min <= state of charge <= soc_max
                    model.addConstr(soc[device][t] <= x[device][t] * devs[device]["max_cap"])
                    model.addConstr(soc[device][t] >= x[device][t] * devs[device]["min_cap"])
                    
                    # charging power <= maximum charging power and discharging power  
                    model.addConstr(ch[device][t-1] <= x[device][t] * devs[device]["max_ch"])
                    model.addConstr(dch[device][t-1] <= x[device][t] * devs[device]["max_dch"])
                    
            
   
    
    #%% ENERGY BALANCES
    for d in days:
        
        for t in time_steps:
 
            if param["switch_single_balance"]:
                # Thermal balance (combined heating and cooling balance)
                model.addConstr(heat["BOI"][d][t] + heat["CHP"][d][t] + heat["HP"][d][t] + heat["EH"][d][t] + dch["TES"][d][t] - cool["AC"][d][t] - cool["CC"][d][t] - cool["air_cooler"][d][t] - dch["CTES"][d][t] == residual["thermal"][d][t]  + heat["AC"][d][t] + ch["TES"][d][t] - ch["CTES"][d][t])
            
            else: # Seperated heating and cooling balance
            
                # Heat balance
                model.addConstr(heat["BOI"][d][t] + heat["CHP"][d][t] + heat["HP"][d][t] + heat["EH"][d][t] + dch["TES"][d][t] == residual["heat"][d][t] + heat["AC"][d][t] + ch["TES"][d][t] )
    
                # Cooling balance
                model.addConstr(cool["AC"][d][t] + cool["CC"][d][t] + cool["air_cooler"][d][t] + dch["CTES"][d][t] == residual["cool"][d][t] + ch["CTES"][d][t] ) 
                
                
            
            # Electricity balance
            model.addConstr(power["CHP"][d][t] + power["PV"][d][t] + power["from_grid"][d][t] == residual["power"][d][t] + power["to_grid"][d][t] + power["CC"][d][t] + power["HP"][d][t] + power["EH"][d][t] )
            
            # Absorption chiller and heat storage can only be supplied by Boiler, CHP and Electic Heater
            model.addConstr(heat["BOI"][d][t] + heat["CHP"][d][t] + heat["EH"][d][t] >= heat["AC"][d][t] + ch["TES"][d][t])
            
            # Cold thermal storage can only be suppled by compression chiller and absorption chiller
            model.addConstr(cool["CC"][d][t] + cool["AC"][d][t] >= ch["CTES"][d][t])    
        
        
        
    #%% DEVICE RESTRICTIONS
    
    if not param["feasible_TES"]:
        model.addConstr(cap["TES"] == 0)
        model.addConstr(cap["CTES"] == 0)
        for d in days:
            for t in time_steps:
                model.addConstr(ch["TES"][d][t] == 0)
                model.addConstr(dch["TES"][d][t] == 0)
                model.addConstr(ch["CTES"][d][t] == 0)
                model.addConstr(dch["CTES"][d][t] == 0)            
        
    if not param["feasible_air_cooler"]: 
        model.addConstr(cap["air_cooler"] == 0)
    
    if not param["feasible_BOI"]: 
        model.addConstr(cap["BOI"] == 0)   
        
    if not param["feasible_CHP"]: 
        model.addConstr(cap["CHP"] == 0) 

    if not param["feasible_AC"]: 
        model.addConstr(cap["AC"] == 0)

    if not param["feasible_CC"]: 
        model.addConstr(cap["CC"] == 0) 

    if not param["feasible_HP"]: 
        model.addConstr(cap["HP"] == 0)  

    if not param["feasible_EH"]: 
        model.addConstr(cap["EH"] == 0)  
        

    #%% SUM UP RESULTS
    
    model.addConstr(gas_total == sum(sum(sum((gas[device][d][t] * param["day_weights"][d]) for t in time_steps) for d in days) for device in ["BOI", "CHP", "to_buildings"]))
  
    model.addConstr(from_grid_total == sum(sum((power["from_grid"][d][t] * param["day_weights"][d]) for t in time_steps) for d in days))
    model.addConstr(to_grid_total == sum(sum((power["to_grid"][d][t] * param["day_weights"][d]) for t in time_steps) for d in days))
    
    model.addConstr(electricity_costs_total == sum(sum((power["from_grid"][d][t] * param["price_el"][d][t] * param["day_weights"][d]) for t in time_steps) for d in days))
    
#    from_DH_total = sum(heat["from_DH"][t] for t in time_steps)
#    from_DC_total = sum(cool["from_DC"][t] for t in time_steps)
    
    # Investment costs
    for device in all_devs:
        model.addConstr( inv[device] == inv_var[device] * cap[device] )
        
    # annualized investment
    for device in all_devs:
        model.addConstr( c_inv[device] == inv[device] * devs[device]["ann_factor"] )

    # Operation and maintenance costs
    for device in all_devs: 
        model.addConstr( c_om[device] == devs[device]["cost_om"] * inv[device] )
    
    
            

    #%% OBJECTIVE
    
    
    model.addConstr(obj ==       sum(c_inv[dev] for dev in all_devs) + sum(c_om[dev] for dev in all_devs)
                                    + sum(tac_building[n] for n in nodes)   
                                    + gas_total * param["price_gas"] + grid_limit_gas * param["price_cap_gas"]
                                    + electricity_costs_total + grid_limit_el * param["price_cap_el"]               # electricity purchase costs + capacity price for grid usage
                                    - to_grid_total * param["revenue_feed_in"]
                                    , "sum_up_TAC")
                                    
        
            
        

#%%
 # Set model parameters and execute calculation
    
    print("Precalculation and model set up done in %f seconds." %(time.time() - start_time))
    
    # Set solver parameters
    model.Params.MIPGap     = 0.01                   # ---,         gap for branch-and-bound algorithm
    model.Params.method     = 2                      # ---,         -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc.
#    model.Params.Heuristics = 0
#    model.Params.MIPFocus   = 2
#    model.Params.Cuts       = 3
#    model.Params.PrePasses  = 8
#    model.Params.Crossover  = 0
#    model.Params.Presolve = 0
    
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
    
        # analyze time series of storage soc's
        if param["feasible_TES"] and param["switch_combined_storage"] and step == 1:
            
            devs["TES"]["x"] = np.zeros(8761)
            devs["CTES"]["x"] = np.zeros(8761)
            
            param["storage_intervals"] = {}
            
                  
            # divide time steps into equal intervals according to number of allowed storage function changes
            if param["n_storage_changes"] == 0:
                param["n_storage_changes"] = 1       
            dt = 8760 / param["n_storage_changes"]
            
            
            for i in range(param["n_storage_changes"]):
                
                if i == 0:
                    t_start = 0
                    t_end = np.int(dt)
                else:
                    t_start = t_end + 1
                    t_end = np.int(dt*(i+1))                   
                    
                # find needed storage function in each interval
                interval = range(t_start, t_end + 1)
                param["storage_intervals"][i] = interval
                hot_mean = np.sum(soc["TES"][t].X for t in interval )  / (t_end - t_start + 1)    # mean soc of hot storage
                cold_mean = np.sum(soc["CTES"][t].X for t in interval ) / (t_end - t_start + 1)   # mean soc of cold storage                
                
                for t in interval:               
                    if hot_mean >= cold_mean:
                        devs["TES"]["x"][t] = 1
                        devs["CTES"]["x"][t] = 0
                    else:
                        devs["TES"]["x"][t] = 0
                        devs["CTES"]["x"][t] = 1  
                        
                # If storage changes its function, it must be empty at the beginning of the new interval
                if i > 0 and devs["TES"]["x"][t_start - 1] != devs["TES"]["x"][t_start]:
                        devs["TES"]["x"][t_start] = 0
                        devs["CTES"]["x"][t_start] = 0
            
            if devs["TES"]["x"][8760] != devs["TES"]["x"][0]:
                        devs["TES"]["x"][0] = 0
                        devs["CTES"]["x"][0] = 0        
                        devs["TES"]["x"][8760] = 0
                        devs["CTES"]["x"][8760] = 0 
                                                
            return devs
    
    
    
        else:
            
            # Write Gurobi files
            model.write(dir_results + "\model.lp")
            model.write(dir_results + "\model.prm")
            model.write(dir_results + "\model.sol")
            
            # Save device capacities in nodes 
            for n in nodes:
                nodes[n]["PV_capacity"] = cap_dom["PV"][n].X
                nodes[n]["HP_capacity"] = cap_dom["HP"][n].X
                nodes[n]["CC_capacity"] = cap_dom["CC"][n].X
                nodes[n]["EH_capacity"] = cap_dom["EH"][n].X
                nodes[n]["BOI_capacity"] = cap_dom["BOI"][n].X
                nodes[n]["TES_capacity"] = cap_dom["TES"][n].X
                nodes[n]["air_cooler_capacity"] = cap_dom["air_cooler"][n].X
                nodes[n]["free_cooler_capacity"] = cap_dom["free_cooler"][n].X
                
                # save residual loads in nodes
                nodes[n]["res_heat_dem"] = np.zeros((param["n_clusters"], 24))
                nodes[n]["power_dem"] = np.zeros((param["n_clusters"], 24))
                for d in days:
                    for t in time_steps:
                        nodes[n]["res_heat_dem"][d][t] = res_thermal[n][d][t].X
                        nodes[n]["power_dem"][d][t] = res_el[n][d][t].X       
            
                # Mass flow from hot to cold pipe
                mass_flow = nodes[n]["res_heat_dem"] * 1000 / (param["c_f"] * (param["T_hot"] - param["T_cold"]))     # kg/s
                nodes[n]["mass_flow"] = mass_flow
                
            
                nodes[n]["tac_building"] = tac_building[n].X 
            
            
            # save annualized costs for devices and gas demand for buildings
            param["tac_buildings"] = sum(nodes[n]["tac_building"] for n in nodes)                                       # kEUR/a, annualized costs for building devices
            param["gas_buildings"] = {}
            for d in days:
                param["gas_buildings"][d] = {}
                for t in time_steps:
                    param["gas_buildings"][d][t] = gas["to_buildings"][d][t].X 
            param["gas_buildings_total"] = sum(sum(param["gas_buildings"][d][t] for t in time_steps) for d in days)
#            param["gas_buildings_max"] = max( sum(gas_dom["BOI"][n][t].X for n in nodes) for t in time_steps) / 1000    # MW, maximum gas load of buildings
            
            
            # Print tac
            print("tac: " + str(obj.X))
            
    
            # Run Post Processing
           # post.run(dir_results, param, nodes)
            
            # Building plots
 #           for n in nodes:
                
            
            # Plot heating and cooling balance for every type-day and every building
            
            
            
            
            # return nodes, param
            return nodes, param





          
                
    
    