# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 23:17:16 2018

@author: lkivi
"""

import numpy as np
import gurobipy as gp
import time
import os
      
def run_optim(nodes, param, devs, devs_dom, dir_results):
    
    time_steps = range(8760)
    t_air = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(0))          # Air temperature °C
           
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    start_time = time.time()

    # Create set for devices
    all_devs = ["BOI", "CHP", "AC", "CC", "EH"] 
    all_devs_dom = ["HP", "CC", "EH", "free_cooler", "air_cooler"]       
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("Global_Optimization")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables
    
    #%% BALANCING UNIT VARIABLES
    
    # Piece-wise linear function variables
    lin = {}
    for device in ["BOI", "CHP", "AC", "CC", "EH"]:   
        lin[device] = {}
        for i in range(len(devs[device]["cap_i"])):
            lin[device][i] = model.addVar(vtype="C", name="lin_" + device + "_i" + str(i))
    
    # Purchase decision binary variables (1 if device is installed, 0 otherwise)
#    x = {}
#    for device in all_devs:
#        x[device] = model.addVar(vtype="B", name="x_" + str(device))
            
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["BOI", "CHP", "AC", "CC", "EH"]:
        cap[device] = model.addVar(vtype="C", name="nominal_capacity_" + str(device))
    
    # Gas flow to/from devices
    gas = {}
    for device in ["BOI", "CHP"]:
        gas[device] = {}
        for t in time_steps:
            gas[device][t] = model.addVar(vtype="C", name="gas_" + device + "_t" + str(t))
        
    # Eletrical power to/from devices
    power = {}
    for device in ["CHP", "CC", "EH", "from_grid", "to_grid"]:
        power[device] = {}
        for t in time_steps:
            power[device][t] = model.addVar(vtype="C", name="power_" + device + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["BOI", "CHP", "AC", "EH"]:
        heat[device] = {}
        for t in time_steps:
            heat[device][t] = model.addVar(vtype="C", name="heat_" + device + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "AC"]:
        cool[device] = {}
        for t in time_steps:
            cool[device][t] = model.addVar(vtype="C", name="cool_" + device + "_t" + str(t))
            
    # grid maximum transmission power
    grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
    grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")
    
    # total energy amounts taken from grid and fed into grid
    from_grid_total = model.addVar(vtype = "C", name="from_grid_total")
    to_grid_total = model.addVar(vtype = "C", name="to_grid_total")
    gas_total = model.addVar(vtype = "C", name="gas_total")

#    # Storage decision variables
#    ch = {}  # Energy flow to charge storage device
#    dch = {} # Energy flow to discharge storage device
#    soc = {} # State of charge
#
#    for device in ["TES", "CTES", "BAT"]:
#        ch[device] = {}
#        dch[device] = {}
#        soc[device] = {}
#        for t in time_steps:
#            ch[device][t] = model.addVar(vtype="C", name="ch_" + device + "_t" + str(t))
#            dch[device][t] = model.addVar(vtype="C", name="dch_" + device + "_t" + str(t))
#            soc[device][t] = model.addVar(vtype="C", name="soc_" + device + "_t" + str(t))
#        soc[device][len(time_steps)] = model.addVar(vtype="C", name="soc_" + device + "_t" + str(len(time_steps)))

    
    
    #%% BUILDING VARIABLES
    
    # Device's capacity (i.e. nominal power)
    cap_dom = {}
    for device in ["HP", "CC", "EH", "free_cooler", "air_cooler"]:
        cap_dom[device] = {}
        for n in nodes:
            cap_dom[device][n] = model.addVar(vtype="C", name="nominal_capacity_" + str(device) + "_n" + str(n))
            
     # Eletrical power to/from devices
    power_dom = {}
    for device in ["HP", "CC", "EH"]:
        power_dom[device] = {}
        for n in nodes:
            power_dom[device][n] = {}
            for t in time_steps:
                power_dom[device][n][t] = model.addVar(vtype="C", name="power_" + device + "_n" + str(n) + "_t" + str(t))
       
    # Heat to/from devices
    heat_dom = {}
    for device in ["HP", "EH"]:
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

    
    # Total residual network load
    residual = {}  
    residual["power"] = {}
    residual["heat"] = {}
    residual["cool"] = {}
    for t in time_steps:        
        residual["power"][t] = model.addVar(vtype = "C", name="residual_power_t" + str(t))
        residual["heat"][t] = model.addVar(vtype = "C", name="residual_heating_t" + str(t))   
        residual["cool"][t] = model.addVar(vtype = "C", name="residual_cooling_t" + str(t))  

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
    
    #%% LOAD CONSTRAINTS
    
    for n in nodes:  
        for t in time_steps:
            for device in ["HP", "EH"]:
                model.addConstr(heat_dom[device][n][t] <= cap_dom[device][n])
            
            for device in ["CC", "free_cooler", "air_cooler"]:
                model.addConstr(cool_dom[device][n][t] <= cap_dom[device][n])
    
    #%% INPUT / OUTPUT CONSTRAINTS
    
    for n in nodes:
        for t in time_steps:
    
            # Electric heater
            model.addConstr(heat_dom["EH"][n][t] == power_dom["EH"][n][t] * devs_dom["EH"]["eta_th"])
            
            # Compression chiller
            model.addConstr(cool_dom["CC"][n][t] == power_dom["CC"][n][t] * devs_dom["CC"]["COP"][n][t])  
    
            # Heat Pump
            model.addConstr(heat_dom["HP"][n][t] == power_dom["HP"][n][t] * devs_dom["HP"]["COP"][n][t])
                
        

    #%% ENERGY BALANCES
            
    for n in nodes:   
        for t in time_steps:
            
            # Heat balance
            model.addConstr(heat_dom["EH"][n][t] + heat_dom["HP"][n][t]  == nodes[n]["heat"][t] )  
    
            # Cooling balance
            model.addConstr(cool_dom["CC"][n][t] + cool_dom["free_cooler"][n][t] + cool_dom["air_cooler"][n][t] == nodes[n]["cool"][t] ) 
            
            # Electricity balance
            model.addConstr(res_el[n][t] == power_dom["EH"][n][t] + power_dom["HP"][n][t] + power_dom["CC"][n][t] )
            

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
                model.addConstr(cool_dom["air_cooler"][n][t] <= m_air[n][t] * param["c_f"]*(nodes[n]["T_cooling_return"][t] - (t_air[t] + devs_dom["air_cooler"]["dT_min"])))
            
            # free cooling
            if param["T_hot"][t] + devs_dom["free_cooler"]["dT_min"] > nodes[n]["T_cooling_return"][t]:
                model.addConstr(m_free[n][t] == 0)
                model.addConstr(cool_dom["free_cooler"][n][t] == 0)
            else:
                model.addConstr(cool_dom["free_cooler"][n][t] <= m_free[n][t] * param["c_f"]*(nodes[n]["T_cooling_return"][t] - (param["T_cold"][t] + devs_dom["free_cooler"]["dT_min"])))
            

    #%% RESIDUAL THERMAL LOADS
    
    for n in nodes:
        for t in time_steps:
            
            model.addConstr(res_thermal[n][t] == (heat_dom["HP"][n][t] - power_dom["HP"][n][t]) - (cool_dom["CC"][n][t] + power_dom["CC"][n][t] + cool_dom["free_cooler"][n][t] ))
            
            model.addConstr(res_thermal_abs[n][t] == gp.abs_(res_thermal[n][t]))
            
            model.addConstr(res_heat[n][t] == (res_thermal_abs[n][t] + res_thermal[n][t]) / 2 )
            model.addConstr(res_cool[n][t] == (res_thermal_abs[n][t] - res_thermal[n][t]) / 2 )
        
    
    #%% DEVICE RESTRICTIONS
    
    if param["use_eh_in_bldgs"] == 0:       
        for n in nodes:  
            model.addConstr(cap_dom["EH"][n] == 0 )
            
    #%% SUM UP

    # Investment costs
    inv_dom = {}
    c_inv_dom = {}
    c_om_dom = {}
 
    for device in all_devs_dom:
        inv_dom[device] = {}
        c_inv_dom[device] = {}
        c_om_dom[device] = {}

        for n in nodes:
            
            # investment costs
            inv_dom[device][n] = devs_dom[device]["inv_var"] * cap_dom[device][n]
        
            # annualized investment
            c_inv_dom[device][n] = inv_dom[device][n] * devs_dom[device]["ann_factor"]
    
            # Operation and maintenance costs 
            c_om_dom[device][n] = devs_dom[device]["cost_om"] * inv_dom[device][n]
            
    
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

        
        
        
    #%% BALANCING UNIT CONSTRAINTS
    
    
    #%% DEVICE CAPACITIES
    # calculate from piece-wise linear function variables    
    for device in ["BOI", "CHP", "AC", "CC", "EH"]:
    
        model.addConstr(cap[device] == sum(lin[device][i] * devs[device]["cap_i"][i] for i in range(len(devs[device]["cap_i"]))))
        # lin: Special Ordered Sets of type 2 (SOS2 or S2): an ordered set of non-negative variables, of which at most two can be non-zero, and if 
        # two are non-zero these must be consecutive in their ordering. 
        model.addSOS(gp.GRB.SOS_TYPE2, [lin[device][i] for i in range(len(devs[device]["cap_i"]))])
        
        # Sum of linear function variables should be 1
        model.addConstr(1 == sum(lin[device][i] for i in range(len(devs[device]["cap_i"]))))  
        
   
    #%% CONTINUOUS SIZING OF DEVICES: minimum capacity <= capacity <= maximum capacity
    
#    for device in ["TES", "CTES", "BAT"]:
#        model.addConstr(cap[device] <= x[device] * devs[device]["max_cap"])
    
    for t in time_steps:
        for device in ["BOI", "EH"]:
            model.addConstr(heat[device][t] <= cap[device])
            
        for device in ["CHP"]:
            model.addConstr(power[device][t] <= cap[device])
        
        for device in ["CC", "AC"]:
            model.addConstr(cool[device][t] <= cap[device])

        # limitation of power from and to grid   
        model.addConstr(sum(gas[device][t] for device in ["BOI", "CHP"]) <= grid_limit_gas)       
        for device in ["from_grid", "to_grid"]:
            model.addConstr(power[device][t] <= grid_limit_el)
            
            
    #%% INPUT / OUTPUT CONSTRAINTS
    for t in time_steps:
        # Boiler
        model.addConstr(gas["BOI"][t] == heat["BOI"][t] / devs["BOI"]["eta_th"])
        
        # Combined heat and power
        model.addConstr(power["CHP"][t] == heat["CHP"][t] / devs["CHP"]["eta_th"] * devs["CHP"]["eta_el"])
        model.addConstr(gas["CHP"][t] == heat["CHP"][t] / devs["CHP"]["eta_th"])
        
        # Electric heater
        model.addConstr(heat["EH"][t] == power["EH"][t] * devs["EH"]["eta_th"])
        
        # Compression chiller
        model.addConstr(cool["CC"][t] == power["CC"][t] * devs["CC"]["COP"][t])  

        # Absorption chiller
        model.addConstr(cool["AC"][t] == heat["AC"][t] * devs["AC"]["eta_th"])

   
    
    #%% ENERGY BALANCES
    for t in time_steps:
        # Heat balance
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] + heat["EH"][t] == residual["heat"][t] + heat["AC"][t] )

    for t in time_steps:
        # Electricity balance
        model.addConstr(power["CHP"][t] + power["from_grid"][t] == residual["power"][t] + power["EH"][t] + power["to_grid"][t] + power["CC"][t] )

    for t in time_steps:
        # Cooling balance
        model.addConstr(cool["AC"][t] + cool["CC"][t] == residual["cool"][t] ) 
        
    # Absorption chiller can only be supplied by Boiler and CHP
    for t in time_steps:
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] >= heat["AC"][t]) 
    
    
    

    #%% SUM UP RESULTS
    model.addConstr(gas_total == sum(sum(gas[device][t] for t in time_steps) for device in ["BOI", "CHP"]))
  
    model.addConstr(from_grid_total == sum(power["from_grid"][t] for t in time_steps))
    model.addConstr(to_grid_total == sum(power["to_grid"][t] for t in time_steps))
#    from_DH_total = sum(heat["from_DH"][t] for t in time_steps)
#    from_DC_total = sum(cool["from_DC"][t] for t in time_steps)
    
    # Investment costs
    inv = {}
    for device in all_devs:
        inv[device] = sum(lin[device][i] * devs[device]["inv_i"][i] for i in range(len(devs[device]["cap_i"]))) 
    
    # annualized investment
    c_inv = {}
    for device in all_devs:
        c_inv[device] = inv[device] * devs[device]["ann_factor"]

    # Operation and maintenance costs
    c_om = {}
    for device in all_devs: 
        c_om[device] = devs[device]["cost_om"] * inv[device]
    
    
            

    #%% OBJECTIVE
    
    
    model.addConstr(obj ==       sum(c_inv[dev] for dev in all_devs) + sum(c_om[dev] for dev in all_devs)
                                    + sum(tac_building[n] for n in nodes)   
                                    + gas_total * param["price_gas"] + grid_limit_gas * param["price_cap_gas"]
                                    + from_grid_total * param["price_el"] + grid_limit_el * param["price_cap_el"]     # grid electricity costs
                                    - to_grid_total * param["revenue_feed_in"]
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
            nodes[n]["HP_capacity"] = cap_dom["HP"][n].X
            nodes[n]["CC_capacity"] = cap_dom["CC"][n].X
            nodes[n]["EH_capacity"] = cap_dom["EH"][n].X
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
        
        
        # save annualized costs for devices
        param["tac_buildings"] = sum(nodes[n]["tac_building"] for n in nodes)          # kEUR/a, annualized costs for building devices
        
        
        # Print tac
        print("tac: " + str(obj.X))
        

    return nodes, param





          
                
    
    