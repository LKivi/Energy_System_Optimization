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
    
    if param["feasible_TES"] and param["switch_combined_storage"]:
        
        devs = run_optim(nodes, param, devs, devs_dom, dir_results, 1)
        
        nodes, param = run_optim(nodes, param, devs, devs_dom, dir_results, 2)
        
    else:
        
        nodes, param = run_optim(nodes, param, devs, devs_dom, dir_results, 1)
        
    return nodes, param
        
        
        
        

#%%
# consider integrated hot and cold storage
def run_optim(nodes, param, devs, devs_dom, dir_results, step):
    
    time_steps = range(8760)
    t_air = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(0))          # Air temperature °C           
    G_sol = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(3))          # Solar radiation W/m^2              
    
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
#    lin = {}
#    for device in ["BOI", "CHP", "AC", "CC", "EH"]:   
#        lin[device] = {}
#        for i in range(len(devs[device]["cap_i"])):
#            lin[device][i] = model.addVar(vtype="C", name="lin_" + device + "_i" + str(i))
    
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
        for t in time_steps:
            gas[device][t] = model.addVar(vtype="C", name="gas_" + device + "_t" + str(t))
        
    # Eletrical power to/from devices
    power = {}
    for device in ["CHP", "CC", "from_grid", "to_grid", "HP", "EH"]:
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
    for device in ["CC", "AC", "air_cooler"]:
        cool[device] = {}
        for t in time_steps:
            cool[device][t] = model.addVar(vtype="C", name="cool_" + device + "_t" + str(t))
            
    # grid maximum transmission power
    grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
    grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")
    
    # total energy amounts taken from grid and fed into grid
    from_grid_total = model.addVar(vtype = "C", name="from_grid_total")
    to_grid_BU = model.addVar(vtype = "C", name="to_grid_BU")
    # PV-generated power to grid
    to_grid_buildings = model.addVar(vtype = "C", name="to_grid_buildings")
    # total power to grid
    to_grid_total = model.addVar(vtype = "C", name="to_grid_total")
    gas_total = model.addVar(vtype = "C", name="gas_total")

    # Storage variables
      
    ch = {}  # Energy flow to charge storage device
    dch = {} # Energy flow to discharge storage device
    soc = {} # State of charge
#    x = {}   # decision varaible determining the function of the storage (hot/cold)

    for device in ["TES", "CTES"]:
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
    for device in ["HP", "CC", "EH", "PV", "to_grid"]:
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
    for device in ["CC", "free_cooler", "air_cooler"]:
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
    air_cooler_max = {}
    for n in nodes:
        m_cooling[n] = {}
        m_free[n] = {}
        m_air[n] = {}
        m_rest[n] = {}
        air_cooler_max[n] = {}
        for t in time_steps:
            m_cooling[n][t] = model.addVar(vtype = "C", name="mass_flow_cooling_n" + str(n) + "_t" + str(t))
        for t in time_steps:
            m_free[n][t] = model.addVar(vtype = "C", name="mass_flow_free_cooler_n" + str(n) + "_t" + str(t))
        for t in time_steps:
            m_air[n][t] = model.addVar(vtype = "C", name="mass_flow_air_cooler_n" + str(n) + "_t" + str(t))
        for t in time_steps:
            m_rest[n][t] = model.addVar(vtype = "C", name="mass_flow_rest_n" + str(n) + "_t" + str(t))
        for t in time_steps:
            air_cooler_max[n][t] = model.addVar(vtype = "C", name="blablablab" + str(n) + "_t" + str(t))

                    
    
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

    
    # Total residual network load
    residual = {}  
    residual["power"] = {}
    residual["heat"] = {}
    residual["cool"] = {}
    for t in time_steps:        
        residual["power"][t] = model.addVar(vtype = "C", name="residual_power_t" + str(t))
    for t in time_steps:
        residual["heat"][t] = model.addVar(vtype = "C", name="residual_heating_t" + str(t))   
    for t in time_steps:
        residual["cool"][t] = model.addVar(vtype = "C", name="residual_cooling_t" + str(t))
        
        
        
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
            c_inv_dom[device][n] = model.addVar(vtype="C", name="annual_inv_costs_" + device + "_n" + str(n))
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
                    model.addConstr(soc_dom[device][n][t] <= devs_dom[device]["soc_max"] * cap_dom[device][n])
                    model.addConstr(soc_dom[device][n][t] >= devs_dom[device]["soc_min"] * cap_dom[device][n])  
    
    for n in nodes:  
        for t in time_steps:
            for device in ["PV"]:
                model.addConstr(power_dom[device][n][t] <= cap_dom[device][n])                
            
            for device in ["HP", "EH", "BOI"]:
                model.addConstr(heat_dom[device][n][t] <= cap_dom[device][n])
            
            for device in ["CC", "free_cooler", "air_cooler"]:
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
            model.addConstr(power_dom["PV"][n][t] <= G_sol[t] * devs_dom["PV"]["eta_el"] * area_dom["PV"][n])            
                
        

    #%% ENERGY BALANCES
            
    for n in nodes:   
        for t in time_steps:
            
            # Heat balance
            model.addConstr(heat_dom["EH"][n][t] + heat_dom["HP"][n][t] + heat_dom["BOI"][n][t] + dch_dom["TES"][n][t]  == nodes[n]["heat"][t] + ch_dom["TES"][n][t] )  
    
            # Cooling balance
            model.addConstr(cool_dom["CC"][n][t] + cool_dom["free_cooler"][n][t] + cool_dom["air_cooler"][n][t] == nodes[n]["cool"][t] ) 
            
            # Electricity balance
            model.addConstr( power_dom["PV"][n][t] + res_el[n][t] == power_dom["EH"][n][t] + power_dom["HP"][n][t] + power_dom["CC"][n][t] + power_dom["to_grid"][n][t] )
            
    
    #%% BUILDING THERMAL STORAGES

    for n in nodes:
    
        for device in ["TES"]:
            
            # Cyclic condition
            model.addConstr(soc_dom[device][n][len(time_steps)] == soc_dom[device][n][0])
        
            for t in range(len(time_steps)+1):
                
                if t == 0:
                    # Set initial state of charge
                    model.addConstr(soc_dom[device][n][0] <= cap_dom[device][n] * devs_dom[device]["soc_init"])
                else:
                    # Energy balance: soc(t) = soc(t-1) + charge - discharge
                    model.addConstr(soc_dom[device][n][t] == soc_dom[device][n][t-1] * (1-devs_dom[device]["sto_loss"])
                        + (ch_dom[device][n][t-1] * devs_dom[device]["eta_ch"] 
                        - dch_dom[device][n][t-1] / devs_dom[device]["eta_dch"]))
                    
                    # charging power <= maximum charging power and discharging power <= maximum discharging power 
                    model.addConstr(ch_dom[device][n][t-1] <= devs_dom[device]["max_ch"])
                    model.addConstr(dch_dom[device][n][t-1] <= devs_dom[device]["max_dch"])   
            

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
                model.addConstr(cool_dom["air_cooler"][n][t] == 0)
            else:
                model.addConstr(cool_dom["air_cooler"][n][t] == m_air[n][t] * param["c_f"] * (nodes[n]["T_cooling_return"][t] - (t_air[t] + devs_dom["air_cooler"]["dT_min"])) / 1000 ) 
            
            # free cooling
            if param["T_hot"][t] + devs_dom["free_cooler"]["dT_min"] > nodes[n]["T_cooling_return"][t]:
                model.addConstr(m_free[n][t] == 0)
                model.addConstr(cool_dom["free_cooler"][n][t] == 0)
            else:
                model.addConstr(cool_dom["free_cooler"][n][t] == m_free[n][t] * param["c_f"] * (nodes[n]["T_cooling_return"][t] - (param["T_cold"][t] + devs_dom["free_cooler"]["dT_min"])) / 1000 )
 

           

    #%% RESIDUAL THERMAL LOADS
    
    for n in nodes:
        for t in time_steps:
            
            model.addConstr(res_thermal[n][t] == (heat_dom["HP"][n][t] - power_dom["HP"][n][t]) - (cool_dom["CC"][n][t] + power_dom["CC"][n][t] + cool_dom["free_cooler"][n][t] ))
            
            model.addGenConstrAbs(res_thermal_abs[n][t], res_thermal[n][t])
            
            model.addConstr(res_heat[n][t] == (res_thermal_abs[n][t] + res_thermal[n][t]) / 2 )
            model.addConstr(res_cool[n][t] == (res_thermal_abs[n][t] - res_thermal[n][t]) / 2 )
        
    
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
            for t in time_steps:
                model.addConstr( ch_dom["TES"][n][t] == 0)
                model.addConstr( dch_dom["TES"][n][t] == 0)
                model.addConstr( soc_dom["TES"][n][t] == 0)
            
            
            
            
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
    for t in time_steps:
        model.addConstr(residual["power"][t] == sum(res_el[n][t] for n in nodes) / 1000)
    for t in time_steps:    
        model.addConstr(residual["heat"][t] == sum(res_heat[n][t] for n in nodes) / 1000)
    for t in time_steps:   
        model.addConstr(residual["cool"][t] == sum(res_cool[n][t] for n in nodes) / 1000)
        
    
    # PV power to grid
    model.addConstr(to_grid_buildings == sum(sum(power_dom["to_grid"][n][t] for n in nodes) for t in time_steps) / 1000 )
        
                
    # Gas supply for building boilers
    for t in time_steps:
        model.addConstr(gas["to_buildings"][t] == sum(gas_dom["BOI"][n][t] for n in nodes) / 1000)

        
        
        
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
        for t in time_steps:    
                model.addConstr(soc[device][t] <= devs[device]["soc_max"] * cap[device])
                model.addConstr(soc[device][t] >= devs[device]["soc_min"] * cap[device])         
        
    for t in time_steps:
        for device in ["BOI", "HP", "EH"]:
            model.addConstr(heat[device][t] <= cap[device])
            
        for device in ["CHP"]:
            model.addConstr(power[device][t] <= cap[device])
        
        for device in ["CC", "AC", "air_cooler"]:
            model.addConstr(cool[device][t] <= cap[device])

        # limitation of power from and to grid   
        model.addConstr(sum(gas[device][t] for device in ["BOI", "CHP"]) + sum(gas_dom["BOI"][n][t] for n in nodes) / 1000 <= grid_limit_gas)       
        for device in ["from_grid", "to_grid"]:
            model.addConstr(power[device][t] <= grid_limit_el)
            
    # Air cooler temperature constraints
    for t in time_steps:
        if t_air[t] + devs_dom["air_cooler"]["dT_min"] > param["T_hot"][t]:
            model.addConstr(cool["air_cooler"][t] == 0)
        else:
            model.addConstr(cool["air_cooler"][t] <= residual["cool"][t] * (param["T_hot"][t] - (t_air[t] + devs_dom["air_cooler"]["dT_min"]))/ (param["T_hot"][t] - param["T_cold"][t] ))
            
            
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
    
    
        if step == 1:
    
            for t in range(len(time_steps)+1):
                
                if t == 0:
                    # Set initial state of charge
                    model.addConstr(soc[device][0] <= cap[device] * devs[device]["soc_init"])
                else:
                    # Energy balance: soc(t) = soc(t-1) + charge - discharge
                    model.addConstr(soc[device][t] == soc[device][t-1] * (1-devs[device]["sto_loss"])
                        + (ch[device][t-1] * devs[device]["eta_ch"] 
                        - dch[device][t-1] / devs[device]["eta_dch"]))
                    
                    # charging power <= maximum charging power and discharging power <= maximum discharging power 
                    model.addConstr(ch[device][t-1] <= devs[device]["max_ch"])
                    model.addConstr(dch[device][t-1] <= devs[device]["max_dch"])
                    
    
       
        
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
    
    for t in time_steps:
        # Heat balance
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] + heat["HP"][t] + heat["EH"][t] + dch["TES"][t] == residual["heat"][t] + heat["AC"][t] + ch["TES"][t] )

    for t in time_steps:
        # Electricity balance
        model.addConstr(power["CHP"][t] + power["from_grid"][t] == residual["power"][t] + power["to_grid"][t] + power["CC"][t] + power["HP"][t] + power["EH"][t] )

    for t in time_steps:
        # Cooling balance
        model.addConstr(cool["AC"][t] + cool["CC"][t] + cool["air_cooler"][t] + dch["CTES"][t] == residual["cool"][t] + ch["CTES"][t] ) 
        
    # Absorption chiller and heat storage can only be supplied by Boiler, CHP and Electic Heater
    for t in time_steps:
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] + heat["EH"][t] >= heat["AC"][t] + ch["TES"][t])
        
    # Cold thermal storage can only be suppled by compression chiller and absorption chiller
    for t in time_steps:
        model.addConstr(cool["CC"][t] + cool["AC"][t] >= ch["CTES"][t])    
        
        
        
    #%% DEVICE RESTRICTIONS
    
    if not param["feasible_TES"]:
        model.addConstr(cap["TES"] == 0)
        model.addConstr(cap["CTES"] == 0)
        for t in time_steps:
            model.addConstr(ch["TES"][t] == 0)
            model.addConstr(dch["TES"][t] == 0)
            model.addConstr(ch["CTES"][t] == 0)
            model.addConstr(dch["CTES"][t] == 0)            
        
    
    
    

    #%% SUM UP RESULTS
    
    model.addConstr(gas_total == sum(sum(gas[device][t] for t in time_steps) for device in ["BOI", "CHP", "to_buildings"]))
  
    model.addConstr(from_grid_total == sum(power["from_grid"][t] for t in time_steps))
    model.addConstr(to_grid_BU == sum(power["to_grid"][t] for t in time_steps))
    model.addConstr(to_grid_total == to_grid_BU + to_grid_buildings)
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
                                    + from_grid_total * param["price_el"] + grid_limit_el * param["price_cap_el"]     # grid electricity costs
                                    - to_grid_BU * param["revenue_feed_in"]["CHP"]
                                    - to_grid_buildings * param["revenue_feed_in"]["PV"]
                                    , "sum_up_TAC")
                                    
        
            
        

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
    model.Params.Crossover  = 0
    
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
                nodes[n]["res_heat_dem"] = np.zeros(8760)
                nodes[n]["power_dem"] = np.zeros(8760)
                for t in time_steps:
                    nodes[n]["res_heat_dem"][t] = res_thermal[n][t].X
                    nodes[n]["power_dem"][t] = res_el[n][t].X       
            
                # Mass flow from hot to cold pipe
                mass_flow = nodes[n]["res_heat_dem"] * 1000 / (param["c_f"] * (param["T_hot"] - param["T_cold"]))     # kg/s
                nodes[n]["mass_flow"] = mass_flow
                
            
                nodes[n]["tac_building"] = tac_building[n].X 
            
            
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





          
                
    
    