# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20

@author: lkivi
"""

import numpy as np
import gurobipy as gp
import time
import os
      

def run(nodes, param, devs, devs_dom, dir_results):
        
    nodes, param = run_optim(nodes, param, devs, devs_dom, dir_results)
        
    return nodes, param
        
        
        
        

#%%
# consider integrated hot and cold storage
def run_optim(nodes, param, devs, devs_dom, dir_results):
    
    time_steps = range(8760)
    t_air = param["t_air"]          # Air temperature °C           
    G_sol = param["G_sol"]          # Solar radiation W/m^2              
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    start_time = time.time()

    # Create set for BU devices
    all_devs = ["BOI", "CHP", "AC", "CC", "TES", "CTES", "HYB", "HP", "EH", "BAT"] 
    
    
    # Create set for building devices
    all_devs_dom = ["HP", "CC", "EH", "FRC", "AIR", "BOI", "PV", "TES"]     
    
    
    ## Simplification of BU-Optimization
    
    # Get constant investment costs (kEUR / MW)
    inv_var = {}
    inv_var["BOI"] = 67.5
    inv_var["CHP"] = 768
    inv_var["EH"] = 150
    inv_var["AC"] = 525
    inv_var["CC"] = 166
    inv_var["HP"] = 300
    inv_var["TES"] = 7.9             # kEUR/MWh_th
    inv_var["CTES"] = 55.2           # kEUR/MWh_th
    inv_var["HYB"] = 240
    inv_var["BAT"] = 520             # kEUR/MW_th
    
        
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("Global_Optimization")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables
    
    #%% BALANCING UNIT VARIABLES
    
    # Piece-wise linear function variables
    lin = {}
    for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "HP"]:   
        lin[device] = {}
        for i in range(len(devs[device]["cap_i"])):
            lin[device][i] = model.addVar(vtype="C", name="lin_" + device + "_i" + str(i))
    
            
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "HYB", "HP", "EH", "BAT"]:
        cap[device] = model.addVar(vtype="C", name="nominal_capacity_" + str(device))
        
    # Storage volumes
    vol = {}
    for device in ["TES", "CTES"]:
        vol[device] = model.addVar(vtype = "C", name="volume_" + str(device))
    
    
    # Gas flow to/from devices
    gas = {}
    for device in ["BOI", "CHP", "to_buildings"]:
        gas[device] = {}
        for t in time_steps:
            gas[device][t] = model.addVar(vtype="C", name="gas_" + device + "_t" + str(t))
        
    # Eletrical power to/from devices
    power = {}
    for device in ["CHP", "CC", "from_grid", "to_grid", "HP", "EH", "PV"]:
        power[device] = {}
        for t in time_steps:
            power[device][t] = model.addVar(vtype="C", name="power_" + device + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["BOI", "CHP", "AC", "HP", "EH"]:
        heat[device] = {}
        for t in time_steps:
            heat[device][t] = model.addVar(vtype="C", name="heat_" + device + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "AC", "HYB"]:
        cool[device] = {}
        for t in time_steps:
            cool[device][t] = model.addVar(vtype="C", name="cool_" + device + "_t" + str(t))
            
    # Feed-in
    feed_in = {}
    for device in ["CHP", "PV"]:
        feed_in[device] = {}
        for t in time_steps:
            feed_in[device][t] = model.addVar(vtype="C", name="feed_in_"+device+"_t"+str(t))
            
    # grid maximum transmission power
    grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
    grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")
    
    # total energy amounts taken from grid and fed into grid
    from_grid_total = model.addVar(vtype = "C", name="from_grid_total")
    # total power to grid
    to_grid_total = model.addVar(vtype = "C", name="to_grid_total")
    # Gas taken from grid
    gas_total = model.addVar(vtype = "C", name="gas_total")
    # total revenue for feed-in
    revenue_feed_in = {}
    for device in ["CHP", "PV"]:
        revenue_feed_in[device] = model.addVar(vtype="C", name="revenue_feed_in_"+str(device))
    # Electricity costs
    electricity_costs = model.addVar(vtype="C", name="electricity_costs")

    # Storage variables
      
    ch = {}  # Energy flow to charge storage device
    dch = {} # Energy flow to discharge storage device
    soc = {} # State of charge
#    x = {}   # decision varaible determining the function of the storage (hot/cold)

    for device in ["TES", "CTES", "BAT"]:
        ch[device] = {}
        dch[device] = {}
        soc[device] = {}
#        x[device] = {}
        for t in time_steps:
            ch[device][t] = model.addVar(vtype="C", name="ch_" + device + "_t" + str(t))
        for t in time_steps:
            dch[device][t] = model.addVar(vtype="C", name="dch_" + device + "_t" + str(t))
        for t in time_steps:
            soc[device][t] = model.addVar(vtype="C", name="soc_" + device + "_t" + str(t))
        soc[device][len(time_steps)] = model.addVar(vtype="C", name="soc_" + device + "_t" + str(len(time_steps)))
              
#        for i in intervals:
#            x[device][i] = model.addVar(vtype="B", name="storage_is_" + device + "_i" + str(i))
        
    # Investment costs
    inv = {}
    c_inv = {}
    c_om = {}  
    c_total = {}
    for device in all_devs:
        inv[device] = model.addVar(vtype = "C", name = "inv_costs_" + device)
    for device in all_devs:
        c_inv[device] = model.addVar(vtype = "C", name = "annual_inv_costs" + device)
    for device in all_devs:
        c_om[device] = model.addVar(vtype = "C", name = "om_costs" + device)
    for device in all_devs:
        c_total[device] = model.addVar(vtype="C", name = "total_annual_costs_"+device)         
        
    
    
    
    #%% BUILDING VARIABLES
    
    # Device's capacity (i.e. nominal power)
    cap_dom = {}
    for device in ["HP", "CC", "EH", "FRC", "AIR", "BOI", "PV", "TES"]:
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
            for t in time_steps:
                power_dom[device][n][t] = model.addVar(vtype="C", name="power_" + device + "_n" + str(n) + "_t" + str(t))

     # Gas to devices
    gas_dom = {}
    for device in ["BOI"]:
        gas_dom[device] = {}
        for n in nodes:
            gas_dom[device][n] = {}
            for t in time_steps:
                gas_dom[device][n][t] = model.addVar(vtype="C", name="gas_" + device + "_n" + str(n) + "_t" + str(t))
       
    # Heat to/from devices
    heat_dom = {}
    for device in ["HP", "EH", "BOI"]:
       heat_dom[device] = {}
       for n in nodes:
           heat_dom[device][n] = {}       
           for t in time_steps:
               heat_dom[device][n][t] = model.addVar(vtype="C", name="heat_"  + device + "_n" + str(n) + "_t" + str(t))
    
    # Cooling power to/from devices
    cool_dom = {}
    for device in ["CC", "FRC", "AIR"]:
        cool_dom[device] = {}
        for n in nodes:
            cool_dom[device][n] = {}
            for t in time_steps:
                cool_dom[device][n][t] = model.addVar(vtype="C", name="cool_" + device + "_n" + str(n) + "_t" + str(t))
                
    # Storage variables    
    
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
            for t in time_steps:
                ch_dom[device][n][t] = model.addVar(vtype="C", name="ch_" + device + "_n" + str(n) + "_t" + str(t))
            for t in time_steps:
                dch_dom[device][n][t] = model.addVar(vtype="C", name="dch_" + device + "_n" + str(n) + "_t" + str(t))
            for t in time_steps:
                soc_dom[device][n][t] = model.addVar(vtype="C", name="soc_" + device + "_n" + str(n) + "_t" + str(t))
            soc_dom[device][n][len(time_steps)] = model.addVar(vtype="C", name="soc_" + device + "_n" + str(n) + "_t" + str(len(time_steps)))
                
   
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
    for n in nodes:
        res_el[n] = {}
        for t in time_steps:
            res_el[n][t] = model.addVar(vtype="C", name="residual_power_demand_n" + str(n) + "_t" + str(t))
        res_thermal[n] = {}
        for t in time_steps:
            res_thermal[n][t] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="residual_thermal_demand_n" + str(n) + "_t" + str(t))

    
    # Total residual network load
    residual = {}  
    residual["power"] = {}
    residual["heat"] = {}
    residual["cool"] = {}
    residual["thermal"] = {}
    residual["thermal_abs"] = {}
    for t in time_steps:        
        residual["power"][t] = model.addVar(vtype = "C", name="residual_power_t" + str(t))
    for t in time_steps:
        residual["heat"][t] = model.addVar(vtype = "C", name="residual_heating_t" + str(t))   
    for t in time_steps:
        residual["cool"][t] = model.addVar(vtype = "C", name="residual_cooling_t" + str(t))
    for t in time_steps:
        residual["thermal"][t] = model.addVar(vtype = "C", lb=-gp.GRB.INFINITY, name="residual_thermal_t" + str(t))        
    for t in time_steps:
        residual["thermal_abs"][t] = model.addVar(vtype = "C", name="residual_thermal_abs_t" + str(t))        
        
    # Investment costs
    inv_dom = {}
    c_inv_dom = {}
    c_om_dom = {}
    c_total_dom = {}
    for device in all_devs_dom:
        inv_dom[device] = {}
        c_inv_dom[device] = {}
        c_om_dom[device] = {} 
        c_total_dom[device] = {}
        for n in nodes:
            inv_dom[device][n] = model.addVar(vtype="C", name="inv_costs_" + device + "_n" + str(n))
        for n in nodes:
            c_inv_dom[device][n] = model.addVar(vtype="C", name="annual_inv_costs_" + device + "_n" + str(n))
        for n in nodes:
            c_om_dom[device][n]  = model.addVar(vtype="C", name="om_costs_" + device + "_n" + str(n))             
        for n in nodes:
            c_total_dom[device][n] = model.addVar(vtype="C", name="total_annual_costs_" + device + "_n" + str(n))


    # Annualized technical building equipment costs
    tac_building = {}
    for n in nodes:
        tac_building[n] = model.addVar(vtype = "c", name="tac_building_n" + str(n))
              
                
    # Total annualized costs
    tac_total = model.addVar(vtype="C", name="total_annualized_costs")
    
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
            model.addConstr(area_dom[device][n] <= devs_dom[device]["max_area"][n])
            model.addConstr(cap_dom[device][n] == area_dom[device][n] * devs_dom[device]["G_stc"]/1000 * devs_dom[device]["eta_el_stc"])
            
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
            for t in time_steps:    
                    model.addConstr(soc_dom[device][n][t] <= cap_dom[device][n]) 
    
    for n in nodes:  
        for t in time_steps:
#            for device in ["PV"]:
#                model.addConstr(power_dom[device][n][t] <= cap_dom[device][n])                
            
            for device in ["HP", "EH", "BOI"]:
                model.addConstr(heat_dom[device][n][t] <= cap_dom[device][n])
            
            for device in ["CC", "FRC", "AIR"]:
                model.addConstr(cool_dom[device][n][t] <= cap_dom[device][n])
                
            # heat pump is limited by maximum supply temperature
#            for device in ["HP"]:
#                model.addConstr(heat_dom[device][n][t] <= nodes[n]["heat"][t] * (devs_dom["HP"]["T_supply_max"] - nodes[n]["T_heating_return"][t])/(nodes[n]["T_heating_supply"][t] - nodes[n]["T_heating_return"][t]))
    
    #%% INPUT / OUTPUT CONSTRAINTS
    
    for n in nodes:
        for t in time_steps:
    
            # Electric heater
            model.addConstr(heat_dom["EH"][n][t] == power_dom["EH"][n][t] * devs_dom["EH"]["eta_th"])
            
            # Compression chiller
            model.addConstr(cool_dom["CC"][n][t] == power_dom["CC"][n][t] * devs_dom["CC"]["COP"][n][t])  
    
            # Heat Pump
            model.addConstr(heat_dom["HP"][n][t] == power_dom["HP"][n][t] * devs_dom["HP"]["COP"][n][t])
            
            # Boiler
            model.addConstr(heat_dom["BOI"][n][t] == gas_dom["BOI"][n][t] * devs_dom["BOI"]["eta_th"])
            
            # PV
            model.addConstr(power_dom["PV"][n][t] <= G_sol[t]/1000 * devs_dom["PV"]["eta_el"][t] * area_dom["PV"][n])            
                
        

    #%% ENERGY BALANCES
            
    for n in nodes:   
        for t in time_steps:
            
            # Heat balance
            model.addConstr(heat_dom["EH"][n][t] + heat_dom["HP"][n][t] + heat_dom["BOI"][n][t] + dch_dom["TES"][n][t]  == nodes[n]["heat"][t] + ch_dom["TES"][n][t] )  
    
            # Cooling balance
            model.addConstr(cool_dom["CC"][n][t] + cool_dom["FRC"][n][t] + cool_dom["AIR"][n][t] == nodes[n]["cool"][t] ) 
            
            # Electricity demands
            model.addConstr(res_el[n][t] == power_dom["EH"][n][t] + power_dom["HP"][n][t] + power_dom["CC"][n][t])
            
            # Thermal storage can only be supplied by Electric Heater and Boiler
            model.addConstr(heat_dom["EH"][n][t] + heat_dom["BOI"][n][t] >= ch_dom["TES"][n][t])
            
    
    #%% BUILDING THERMAL STORAGES

    for n in nodes:
    
        for device in ["TES"]:
            
            # Cyclic condition
            model.addConstr(soc_dom[device][n][len(time_steps)] == soc_dom[device][n][0])
        
            for t in np.arange(1,len(time_steps)+1):
                # Energy balance: soc(t) = soc(t-1) + charge - discharge
                model.addConstr(soc_dom[device][n][t] == soc_dom[device][n][t-1] * (1-devs_dom[device]["sto_loss"])
                    + (ch_dom[device][n][t-1] * devs_dom[device]["eta_ch"] 
                    - dch_dom[device][n][t-1] / devs_dom[device]["eta_dch"]))
                      
            

    #%% FREE COOLING AND AIR COOLING RESTRICTIONS
    
    for n in nodes:
        for t in time_steps:
            
            # Mass flow in cooling circle
            model.addConstr(m_cooling[n][t] == nodes[n]["cool"][t] / (param["c_f"] * (nodes[n]["T_cooling_return"][t] - nodes[n]["T_cooling_supply"][t])) * 1000)
        
            # Sum of mass flows
            model.addConstr(m_cooling[n][t] == m_air[n][t] + m_free[n][t] + m_rest[n][t])
            
            # air cooling
            if t_air[t] + devs_dom["AIR"]["dT_min"] > nodes[n]["T_cooling_return"][t]:
                model.addConstr(m_air[n][t] == 0)
                model.addConstr(cool_dom["AIR"][n][t] == 0)
            else:
                model.addConstr(cool_dom["AIR"][n][t] == m_air[n][t] * param["c_f"] * (nodes[n]["T_cooling_return"][t] - (t_air[t] + devs_dom["AIR"]["dT_min"])) / 1000 ) 
            
            # free cooling
            if param["T_hot"][t] + devs_dom["FRC"]["dT_min"] > nodes[n]["T_cooling_return"][t]:
                model.addConstr(m_free[n][t] == 0)
                model.addConstr(cool_dom["FRC"][n][t] == 0)
            else:
                model.addConstr(cool_dom["FRC"][n][t] == m_free[n][t] * param["c_f"] * (nodes[n]["T_cooling_return"][t] - (param["T_cold"][t] + devs_dom["FRC"]["dT_min"])) / 1000 )
 

           

    #%% RESIDUAL THERMAL LOADS
    
    for n in nodes:
        for t in time_steps:
            
            model.addConstr(res_thermal[n][t] == (heat_dom["HP"][n][t] - power_dom["HP"][n][t]) - (cool_dom["CC"][n][t] + power_dom["CC"][n][t] + cool_dom["FRC"][n][t] ))
            
        
    
    #%% DEVICE RESTRICTIONS

    if param["use_hp_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["HP"][n] == 0 )

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
            
    if param["use_frc_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["FRC"][n] == 0 )

    if param["use_air_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["AIR"][n] == 0 )
            
    if param["use_tes_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["TES"][n] == 0 )
            for t in time_steps:
                model.addConstr( ch_dom["TES"][n][t] == 0)
                model.addConstr( dch_dom["TES"][n][t] == 0)
                model.addConstr( soc_dom["TES"][n][t] == 0)
            
            
            
            
    #%% SUM UP

    # Investment costs
 
    for device in all_devs_dom:

        for n in nodes:
            
            # investment costs
            model.addConstr( inv_dom[device][n] == devs_dom[device]["inv_var"] * cap_dom[device][n] )
        
            # annualized investment
            model.addConstr( c_inv_dom[device][n] == devs_dom[device]["ann_inv_var"] * cap_dom[device][n] )
    
            # Operation and maintenance costs 
            model.addConstr( c_om_dom[device][n] == devs_dom[device]["cost_om"] * inv_dom[device][n] )
            
            # tac for building device (kEUR)
            model.addConstr( c_total_dom[device][n] == (c_inv_dom[device][n] + c_om_dom[device][n]) / 1000)
             
    
    # Residual loads
    for t in time_steps:
        model.addConstr(residual["power"][t] == sum(res_el[n][t] for n in nodes) / 1000)
    for t in time_steps:
        model.addConstr(residual["thermal"][t] == sum(res_thermal[n][t] for n in nodes) / 1000)

    
    # if heating and cooling balances are considered separately: divide residual thermal load into heating and cooling load
    if not param["switch_single_balance"]:
        for t in time_steps:  
        
            model.addGenConstrAbs(residual["thermal_abs"][t], residual["thermal"][t])
            
            model.addConstr(residual["heat"][t] == (residual["thermal_abs"][t] + residual["thermal"][t]) / 2)
            model.addConstr(residual["cool"][t] == (residual["thermal_abs"][t] - residual["thermal"][t]) / 2)      
                
    # Gas supply for building boilers
    for t in time_steps:
        model.addConstr(gas["to_buildings"][t] == sum(gas_dom["BOI"][n][t] for n in nodes) / 1000)
        
    # PV generation in buildings
    for t in time_steps:
        model.addConstr(power["PV"][t] == sum(power_dom["PV"][n][t] for n in nodes) / 1000)

        
        
        
    #%% BALANCING UNIT CONSTRAINTS
        
   
    #%% CAPACITY CONSTRAINTS
    
    for device in ["TES", "CTES", "BAT"]:
        model.addConstr(cap[device] <= devs[device]["max_cap"])
        model.addConstr(cap[device] >= devs[device]["min_cap"])
    
    for device in ["TES", "CTES"]:
        # Relation between volume and capacity
        model.addConstr(cap[device] == vol[device] * param["rho_f"] *  param["c_f"] * (devs[device]["T_max"] - devs[device]["T_min"]) / (1e6 * 3600))
     
    if param["switch_cost_functions"]:
        # calculate capacities from piece-wise linear variables
        for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "HP"]:
            
            model.addConstr(cap[device] == sum(lin[device][i] * devs[device]["cap_i"][i] for i in range(len(devs[device]["cap_i"]))))
            # lin: Special Ordered Sets of type 2 (SOS2 or S2): an ordered set of non-negative variables, of which at most two can be non-zero, and if 
            # two are non-zero these must be consecutive in their ordering. 
            model.addSOS(gp.GRB.SOS_TYPE2, [lin[device][i] for i in range(len(devs[device]["cap_i"]))])
            
            # Sum of linear function variables should be 1
            model.addConstr(1 == sum(lin[device][i] for i in range(len(devs[device]["cap_i"]))))
              

    #%% LOAD CONSTRAINTS
    
    for device in ["TES", "CTES", "BAT"]:
        for t in time_steps:    
                model.addConstr(soc[device][t] <= cap[device])     
        
    for t in time_steps:
        for device in ["BOI", "HP", "EH"]:
            model.addConstr(heat[device][t] <= cap[device])
            
        for device in ["CHP"]:
            model.addConstr(power[device][t] <= cap[device])
        
        for device in ["CC", "AC", "HYB"]:
            model.addConstr(cool[device][t] <= cap[device])

        # limitation of power from and to grid   
        model.addConstr(sum(gas[device][t] for device in ["BOI", "CHP", "to_buildings"])  <= grid_limit_gas)       
        for device in ["from_grid", "to_grid"]:
            model.addConstr(power[device][t] <= grid_limit_el)
            
    # Hybrid cooler temperature constraints
    for t in time_steps:
        if t_air[t] + devs["HYB"]["dT_min"] > param["T_hot"][t]:
            model.addConstr(cool["HYB"][t] == 0)
        else:
            if param["switch_single_balance"]:
                model.addConstr(cool["HYB"][t] <= ( -residual["thermal"][t] + heat["BOI"][t] + heat["CHP"][t] + heat["HP"][t] + heat["EH"][t] + dch["TES"][t]) * (param["T_hot"][t] - (t_air[t] + devs["HYB"]["dT_min"]))/ (param["T_hot"][t] - param["T_cold"][t] ))
            else:
                model.addConstr(cool["HYB"][t] <= residual["cool"][t] * (param["T_hot"][t] - (t_air[t] + devs["HYB"]["dT_min"]))/ (param["T_hot"][t] - param["T_cold"][t] ))
            
            
    #%% INPUT / OUTPUT CONSTRAINTS
    for t in time_steps:
        # Boiler
        model.addConstr(gas["BOI"][t] == heat["BOI"][t] / devs["BOI"]["eta_th"])
        
        # Heat pump
        model.addConstr(heat["HP"][t] == power["HP"][t] * devs["HP"]["COP"][t])
        
        # Combined heat and power
        model.addConstr(power["CHP"][t] == heat["CHP"][t] / devs["CHP"]["eta_th"] * devs["CHP"]["eta_el"])
        model.addConstr(gas["CHP"][t] == heat["CHP"][t] / devs["CHP"]["eta_th"])
        
#        # Electric heater
        model.addConstr(heat["EH"][t] == power["EH"][t] * devs["EH"]["eta_th"])
        
        # Compression chiller
        model.addConstr(cool["CC"][t] == power["CC"][t] * devs["CC"]["COP"][t])  

        # Absorption chiller
        model.addConstr(cool["AC"][t] == heat["AC"][t] * devs["AC"]["eta_th"])
        
    
     #%% STORAGE DEVICES
    
    
    for device in ["TES", "CTES"]:
        
        # Cyclic condition
        model.addConstr(soc[device][len(time_steps)] == soc[device][0])
    
        for t in np.arange(1,len(time_steps)+1):
            # Energy balance: soc(t) = soc(t-1) + charge - discharge
            model.addConstr(soc[device][t] == soc[device][t-1] * (1-devs[device]["sto_loss"])
                + (ch[device][t-1] * devs[device]["eta_ch"] 
                - dch[device][t-1] / devs[device]["eta_dch"]))
       
    
                    
                    
            
   
    
    #%% ENERGY BALANCES
    
    for t in time_steps:
        
        
        if param["switch_single_balance"]:
            # Thermal balance (combined heating and cooling balance)
            model.addConstr( heat["BOI"][t] + heat["CHP"][t] + heat["HP"][t] + heat["EH"][t] + dch["TES"][t] - cool["AC"][t] - cool["CC"][t] - cool["HYB"][t] - dch["CTES"][t] == residual["thermal"][t] + heat["AC"][t] + ch["TES"][t] - ch["CTES"][t])
        
        else: # Seperated heating and cooling balance
            
            # Heat balance
            model.addConstr(heat["BOI"][t] + heat["CHP"][t] + heat["HP"][t] + heat["EH"][t] + dch["TES"][t] == residual["heat"][t] + heat["AC"][t] + ch["TES"][t] )
    
            # Cooling balance
            model.addConstr(cool["AC"][t] + cool["CC"][t] + cool["HYB"][t] + dch["CTES"][t] == residual["cool"][t] + ch["CTES"][t] ) 




    for t in time_steps:
        # Electricity balance
        model.addConstr(power["CHP"][t] + power["PV"][t] +  power["from_grid"][t] + dch["BAT"][t] == residual["power"][t] + power["to_grid"][t] + power["CC"][t] + power["HP"][t] + power["EH"][t] + ch["BAT"][t] )
        
    # Absorption chiller and heat storage can only be supplied by Boiler, CHP and Electic Heater
    for t in time_steps:
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] + heat["EH"][t] >= heat["AC"][t] + ch["TES"][t])
        
    # Cold thermal storage can only be suppled by compression chiller and absorption chiller
    for t in time_steps:
        model.addConstr(cool["CC"][t] + cool["AC"][t] >= ch["CTES"][t])    
        

    #%% GRID FEED-IN
    
    for t in time_steps:
        
        # Allocate grid-feed-in to CHP and PV
        model.addConstr(power["to_grid"][t] == feed_in["CHP"][t] + feed_in["PV"][t])
        
        model.addConstr(feed_in["CHP"][t] <= power["CHP"][t] + dch["BAT"][t])
        model.addConstr(feed_in["PV"][t] <= power["PV"][t])
        

        
        
    #%% DEVICE RESTRICTIONS
    
    if not param["feasible_TES"]:
        model.addConstr(cap["TES"] == 0)
        for t in time_steps:
            model.addConstr(ch["TES"][t] == 0)
            model.addConstr(dch["TES"][t] == 0)           
        
    if not param["feasible_CTES"]:
        model.addConstr(cap["CTES"] == 0)
        for t in time_steps:
            model.addConstr(ch["CTES"][t] == 0)
            model.addConstr(dch["CTES"][t] == 0) 
            
    if not param["feasible_BAT"]:
        model.addConstr(cap["BAT"] == 0)
        for t in time_steps:
            model.addConstr(ch["BAT"][t] == 0)
            model.addConstr(dch["BAT"][t] == 0)  
            
    if not param["feasible_HYB"]:
        model.addConstr(cap["HYB"] == 0)    

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
    
    model.addConstr(gas_total == sum(sum(gas[device][t] for t in time_steps) for device in ["BOI", "CHP", "to_buildings"]))
  
    model.addConstr(from_grid_total == sum(power["from_grid"][t] for t in time_steps))
    model.addConstr(to_grid_total == sum(power["to_grid"][t] for t in time_steps))
    
    model.addConstr(electricity_costs == sum((power["from_grid"][t] * param["price_el"][t]) for t in time_steps))

    
    for device in ["PV", "CHP"]:
        model.addConstr(revenue_feed_in[device] == sum((feed_in[device][t] * param["revenue_feed_in"][device][t]) for t in time_steps))
    
    # Investment costs
    if param["switch_cost_functions"]:
        for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "HP"]:
            model.addConstr( inv[device] == sum(lin[device][i] * devs[device]["inv_i"][i] for i in range(len(devs[device]["cap_i"]))) )
        for device in ["EH", "HYB"]:
            model.addConstr(inv[device] == devs[device]["inv_var"] * cap[device])            
    else:
        for device in all_devs:
            model.addConstr(inv[device] == inv_var[device] * cap[device])
        
    # annualized investment
    for device in all_devs:
        model.addConstr( c_inv[device] == inv[device] * devs[device]["ann_factor"] )

    # Operation and maintenance costs
    for device in all_devs: 
        model.addConstr( c_om[device] == devs[device]["cost_om"] * inv[device] )
        
    # Annualized costs for device
    for device in all_devs:
        model.addConstr(c_total[device] == c_inv[device] + c_om[device])
    
    
            

    #%% OBJECTIVE
    
    
    model.addConstr(tac_total ==            sum(c_total[dev] for dev in all_devs)
                                    + sum(sum(c_total_dom[dev][n] for n in nodes) for dev in all_devs_dom)   
                                    + gas_total * param["price_gas"] + grid_limit_gas * param["price_cap_gas"]
                                    + electricity_costs + grid_limit_el * param["price_cap_el"]     # grid electricity costs
                                    - sum(revenue_feed_in[dev] for dev in ["CHP", "PV"])
                                    , "sum_up_TAC")
    
    model.addConstr(obj == tac_total)
                                    
        
            
        

#%%
 # Set model parameters and execute calculation
    
    print("Precalculation and model set up done in %f seconds." %(time.time() - start_time))
    
    # Set solver parameters
    model.Params.MIPGap     = 0.01   # ---,         gap for branch-and-bound algorithm
#    model.Params.method     = 2                 # ---,         -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc.
#    model.Params.Heuristics = 0
#    model.Params.MIPFocus   = 2
#    model.Params.Cuts       = 3
#    model.Params.PrePasses  = 8
#    model.Params.Crossover  = 0
    
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
            nodes[n]["PV_capacity"] = cap_dom["PV"][n].X
            nodes[n]["HP_capacity"] = cap_dom["HP"][n].X
            nodes[n]["CC_capacity"] = cap_dom["CC"][n].X
            nodes[n]["EH_capacity"] = cap_dom["EH"][n].X
            nodes[n]["BOI_capacity"] = cap_dom["BOI"][n].X
            nodes[n]["TES_capacity"] = cap_dom["TES"][n].X
            nodes[n]["air_cooler_capacity"] = cap_dom["AIR"][n].X
            nodes[n]["free_cooler_capacity"] = cap_dom["FRC"][n].X
            
            # save residual loads in nodes
            nodes[n]["res_heat_dem"] = np.zeros(8760)
            nodes[n]["power_dem"] = np.zeros(8760)
            for t in time_steps:
                nodes[n]["res_heat_dem"][t] = res_thermal[n][t].X
                nodes[n]["power_dem"][t] = res_el[n][t].X       
        
            # Mass flow from hot to cold pipe
            mass_flow = nodes[n]["res_heat_dem"] * 1000 / (param["c_f"] * (param["T_hot"] - param["T_cold"]))     # kg/s
            nodes[n]["mass_flow"] = mass_flow
            
        
            nodes[n]["tac_building"] = sum(c_total_dom[dev][n].X for dev in all_devs_dom)
        
        
        # save annualized costs for devices and gas demand for buildings
        param["tac_buildings"] = sum(nodes[n]["tac_building"] for n in nodes)                                       # kEUR/a, annualized costs for building devices
        param["gas_buildings"] = {}
        for t in time_steps:
            param["gas_buildings"][t] = gas["to_buildings"][t].X 
        param["gas_buildings_total"] = sum(param["gas_buildings"][t] for t in time_steps)
#            param["gas_buildings_max"] = max( sum(gas_dom["BOI"][n][t].X for n in nodes) for t in time_steps) / 1000    # MW, maximum gas load of buildings
        
        
        # Print tac
        print("tac: " + str(obj.X))
        

        return nodes, param





          
                
    
    