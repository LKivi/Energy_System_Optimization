# -*- coding: utf-8 -*-
"""

Author: Marco Wirtz, Institute for Energy Efficient Buildings and Indoor Climate, RWTH Aachen University, Germany

Created: 01.09.2018

"""

from __future__ import division
import gurobipy as gp
import os
import parameter
import json
import time
import numpy as np

def run_optim(obj_fn, obj_eps, eps_constr, dir_results):

    assert (obj_eps == "" and eps_constr == "") or (obj_eps != "" and eps_constr != ""), "If there is a bounded objective function, an epsilon constraint should be given."
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load model parameter
    start_time = time.time()
    
    (devs, param, dem) = parameter.load_params()
    
          
    time_steps = range(8760)

    # Create set for devices
    all_devs = ["BOI", "CHP", "AC", "CC", "HP"]       
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("DHC_Benchmark")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables

    # Purchase decision binary variables (1 if device is installed, 0 otherwise)
#    x = {}
#    for device in ["TES"]:
#        x[device] = model.addVar(vtype="B", name="x_" + str(device))
    
    # Piece-wise linear function variables
    lin = {}
    for device in ["BOI", "CHP", "AC", "CC", "HP"]:   
        lin[device] = {}
        for i in range(len(devs[device]["cap_i"])):
            lin[device][i] = model.addVar(vtype="C", name="lin_" + device + "_i" + str(i))
            
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["BOI", "CHP", "AC", "CC", "HP"]:
        cap[device] = model.addVar(vtype="C", name="nominal_capacity_" + str(device))
    
    # Gas flow to/from devices
    gas = {}
    for device in ["BOI", "CHP"]:
        gas[device] = {}
        for t in time_steps:
            gas[device][t] = model.addVar(vtype="C", name="gas_" + device + "_t" + str(t))
        
    # Eletrical power to/from devices
    power = {}
    for device in ["CHP", "CC", "HP", "from_grid", "to_grid"]:
        power[device] = {}
        for t in time_steps:
            power[device][t] = model.addVar(vtype="C", name="power_" + device + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["BOI", "CHP", "AC", "HP"]:
        heat[device] = {}
        for t in time_steps:
            heat[device][t] = model.addVar(vtype="C", name="heat_" + device + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "AC", "HP"]:
        cool[device] = {}
        for t in time_steps:
            cool[device][t] = model.addVar(vtype="C", name="cool_" + device + "_t" + str(t))
            
    if param["switch_transient_hp"]:
        dt_hp = {}
        ratio = {}
        for t in time_steps:
            dt_hp[t] = model.addVar(vtype="C", name="dt_hp_t" + str(t))
        for t in time_steps:
            ratio[t] = model.addVar(vtype="C", name="COP_ratio_HP" + str(t))
            
        lin_HP = {}
        for i in range(len(devs["HP"]["dt_h_i"])):
            lin_HP[i] = {}
            for t in time_steps:
                lin_HP[i][t] = model.addVar(vtype="C", name="lin_HP_i" + str(i) + "_" + str(t))
    
        
    # storage variables
#    ch = {}  # Energy flow to charge storage device
#    dch = {} # Energy flow to discharge storage device
#    soc = {} # State of charge   
#    
#    for device in ["TES"]:
#        ch[device] = {}
#        dch[device] = {}
#        soc[device] = {}
#        for t in time_steps:
#            ch[device][t] = model.addVar(vtype="C", name="ch_" + device + "_t" + str(t))
#            dch[device][t] = model.addVar(vtype="C", name="dch_" + device + "_t" + str(t))
#            soc[device][t] = model.addVar(vtype="C", name="soc_" + device + "_t" + str(t))
#        soc[device][len(time_steps)] = model.addVar(vtype="C", name="soc_" + device + "_t" + str(len(time_steps)))
  
    # grid maximum transmission power
    grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
    grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")
    
    # total energy amounts taken from grid and fed into grid
    from_grid_total = model.addVar(vtype = "C", name="from_grid_total")
    to_grid_total = model.addVar(vtype = "C", name="to_grid_total")
    gas_total = model.addVar(vtype = "C", name="gas_total")
    
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Objective functions
    obj = {}
    set_obj = ["tac", "co2_gross", "power_from_grid", "net_power_from_grid"] # Mögliche Zielgrößen
    for k in set_obj:
        obj[k] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="obj_" + k)    
      
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Define objective function

    model.update()
    model.setObjective(obj[obj_fn], gp.GRB.MINIMIZE)
    if obj_eps == "":
        print("-----------\nSingle-objective optimization with objective function: " + obj_fn)
    else:
        if eps_constr >= 0:
            model.addConstr(obj[obj_eps] <= eps_constr * (1 + param["MIPGap"]))
        elif eps_constr < 0:
            model.addConstr(obj[obj_eps] <= eps_constr * (1 - param["MIPGap"]))
        print("-----------\nRun optimization for '" + obj_fn + "'. Epsilon constraint for '" + obj_eps + "': " + str(eps_constr) + ".")


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Add constraints
 
    #%% DEVICE CAPACITIES
   
    # calculate from piece-wise linear function variables    
    for device in ["BOI", "CHP", "AC", "CC", "HP"]:
    
        model.addConstr(cap[device] == sum(lin[device][i] * devs[device]["cap_i"][i] for i in range(len(devs[device]["cap_i"]))))
        # lin: Special Ordered Sets of type 2 (SOS2 or S2): an ordered set of non-negative variables, of which at most two can be non-zero, and if 
        # two are non-zero these must be consecutive in their ordering. 
        model.addSOS(gp.GRB.SOS_TYPE2, [lin[device][i] for i in range(len(devs[device]["cap_i"]))])
        
        # Sum of linear function variables should be 1
        model.addConstr(1 == sum(lin[device][i] for i in range(len(devs[device]["cap_i"]))))
        
      
    #%% CONTINUOUS SIZING OF DEVICES: minimum capacity <= capacity <= maximum capacity
    
#    for device in ["TES"]:
#        model.addConstr(cap[device] <= x[device] * devs[device]["max_cap"])
#        model.addConstr(cap[device] >= x[device] * devs[device]["min_cap"])
    
#    for device in ["CHP"]:
#        model.addConstr(cap[device] <= x_funding * param["limit_CHP_fund"] + (1 - x_funding) * 100)
#       model.addConstr(cap[device] <= param["limit_CHP_fund"])
    
    # if heat pump is not considered
    if param["switch_hp"] == 0:
        model.addConstr(cap["HP"] == 0)
        
    # if storage is not considered
#    if devs["TES"]["switch_TES"] == 0:
#        model.addConstr(x["TES"] == 0)      
#        for t in time_steps:
#            model.addConstr(ch["TES"][t] == 0)
#            model.addConstr(dch["TES"][t] == 0)
    
    #%% LOAD CONTRAINTS: minimal load < load < capacity
    
    for t in time_steps:
        for device in ["BOI", "HP"]:
            model.addConstr(heat[device][t] <= cap[device])
            
        for device in ["CHP"]:
            model.addConstr(power[device][t] <= cap[device])
        
        for device in ["CC", "AC"]:
            model.addConstr(cool[device][t] <= cap[device])
            
        for device in ["HP"]:
            if not param["switch_transient_hp"]:
                model.addConstr(heat[device][t] <= devs["HP"]["dT_cond"]/(param["T_heating_supply"][t] - param["T_heating_return"]) * dem["heat"][t])      # maximum HP heating
            else:
                 model.addConstr( heat["HP"][t] == dem["heat"][t]/(param["T_heating_supply"][t] - param["T_heating_return"]) * dt_hp[t])           
            model.addConstr(cool[device][t] <= devs["HP"]["dT_evap"]/(param["T_cooling_return"] - param["T_cooling_supply"]) * dem["cool"][t])         # maximum HP cooling
            
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
        
        # Compression chiller
        model.addConstr(cool["CC"][t] == power["CC"][t] * devs["CC"]["COP"][t])  

        # Absorption chiller
        model.addConstr(cool["AC"][t] == heat["AC"][t] * devs["AC"]["eta_th"])
        
        # Heat Pump
        model.addConstr(heat["HP"][t] == power["HP"][t] + cool["HP"][t])            # Heat pump energy balance
        
        if not param["switch_transient_hp"]:
            model.addConstr(heat["HP"][t] == power["HP"][t] * devs["HP"]["COP"])        # COP relation
        else:
            model.addConstr(ratio[t] == sum(lin_HP[i][t] * devs["HP"]["ratio_i"][i] for i in range(len(devs["HP"]["dt_h_i"])))) 
            model.addConstr( power["HP"][t] == dem["heat"][t]/(param["T_heating_supply"][t] - param["T_heating_return"]) * ratio[t])
       
        
        
   
    #%% dt_HP
    if param["switch_transient_hp"]:
        for t in time_steps:
            model.addConstr(dt_hp[t] == sum(lin_HP[i][t] * devs["HP"]["dt_h_i"][i] for i in range(len(devs["HP"]["dt_h_i"]))))
            # lin: Special Ordered Sets of type 2 (SOS2 or S2): an ordered set of non-negative variables, of which at most two can be non-zero, and if 
            # two are non-zero these must be consecutive in their ordering. 
            model.addSOS(gp.GRB.SOS_TYPE2, [lin_HP[i][t] for i in range(len(devs["HP"]["dt_h_i"]))])
                
            # Sum of linear function variables should be 1
            model.addConstr(1 == sum(lin_HP[i][t] for i in range(len(devs["HP"]["dt_h_i"])))) 
            
            model.addConstr(dt_hp[t] <= param["T_heating_supply"][t] - param["T_heating_return"])


    #%% GLOBAL ENERGY BALANCES
    for t in time_steps:
        # Heat balance
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] + heat["HP"][t] == dem["heat"][t] + heat["AC"][t])

    for t in time_steps:
        # Electricity balance
        model.addConstr(power["CHP"][t] + power["from_grid"][t] == power["to_grid"][t] + power["CC"][t] + power["HP"][t])

    for t in time_steps:
        # Cooling balance
        model.addConstr(cool["AC"][t] + cool["CC"][t] + cool["HP"][t] == dem["cool"][t])
        
    
    # Absorption chiller can only be supplied by Boiler and CHP
    for t in time_steps:
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] >= heat["AC"][t])    
        
    
    #%% N-1-REDUNDANCE
    
    if param["switch_n_1"] == 1:
        for t in time_steps:
            
            # AC Supply
            model.addConstr(cap["CHP"] / devs["CHP"]["eta_el"] * devs["CHP"]["eta_th"] >= cap["AC"] / devs["AC"]["eta_th"])        
            model.addConstr(cap["BOI"] >= cap["AC"] / devs["AC"]["eta_th"])
            
            # Heat demand supply
            model.addConstr(cap["CHP"] / devs["CHP"]["eta_el"] * devs["CHP"]["eta_th"] + cap["HP"] >= np.max(dem["heat"]))
            model.addConstr(cap["BOI"] + cap["HP"] >= np.max(dem["heat"]))
            model.addConstr(cap["BOI"] + cap["CHP"] / devs["CHP"]["eta_el"] * devs["CHP"]["eta_th"] >= np.max(dem["heat"])) 
            
            # Cooling demand supply
            model.addConstr(cap["AC"] + cap["CC"] >= np.max(dem["cool"]))
            model.addConstr(cap["CC"] + cap["HP"]*(1-1/devs["HP"]["COP"]) >= np.max(dem["cool"]))
            model.addConstr(cap["AC"] + cap["HP"]*(1-1/devs["HP"]["COP"]) >= np.max(dem["cool"]))
            
        
        
        
        
    #%% STORAGE DEVICES
#    for device in ["TES"]:  
#        # Cyclic condition
#        model.addConstr(soc[device][len(time_steps)] == soc[device][0])     
#
#        for t in range(len(time_steps)+1):
#                    
#            if t == 0:
#                # Set initial state of charge
#                model.addConstr(soc[device][0] <= cap[device] * devs[device]["soc_init"])
#            else:
#                # Energy balance: soc(t) = soc(t-1) + charge - discharge
#                model.addConstr(soc[device][t] == soc[device][t-1] * (1-devs[device]["sto_loss"])
#                    + (ch[device][t-1] * devs[device]["eta_ch"] 
#                    - dch[device][t-1] / devs[device]["eta_dch"]))
#                
#                # soc_min <= state of charge <= soc_max
#                model.addConstr(soc[device][t] <= devs[device]["soc_max"] * cap[device])
#                model.addConstr(soc[device][t] >= devs[device]["soc_min"] * cap[device])
#                
#                # charging power <= maximum charging power and discharging power <= maximum discharging power 
#                model.addConstr(ch[device][t-1] <= devs[device]["max_ch"])
#                model.addConstr(dch[device][t-1] <= devs[device]["max_dch"])

    #%% SUM UP RESULTS
    model.addConstr(gas_total == sum(sum(gas[device][t] for t in time_steps) for device in ["BOI", "CHP"]))
  
    model.addConstr(from_grid_total == sum(power["from_grid"][t] for t in time_steps))
    model.addConstr(to_grid_total == sum(power["to_grid"][t] for t in time_steps))

    # total investment costs
    inv = {}
    for device in all_devs:
        inv[device] = sum(lin[device][i] * devs[device]["inv_i"][i] for i in range(len(devs[device]["cap_i"]))) 

    # Annual investment costs
    c_inv = {}
    for device in all_devs:
        c_inv[device] = inv[device] * devs[device]["ann_factor"]
    
    # Operation and maintenance costs
    c_om = {}
    for device in all_devs:       
        c_om[device] = devs[device]["cost_om"] * inv[device]
        
    # Total generated electrial energy
    generation_total = sum(power["CHP"][t] for t in time_steps)
    
#    # funding of CHP-generated power
#    model.addConstr(revenues_funding <= x_funding * param["limit_CHP_fund"]*8760*param["CHP_funding"])
#    model.addConstr(revenues_funding <= to_grid_total * param["CHP_funding"])
    

    #%% OBJECTIVE FUNCTIONS
    # TOTAL ANNUALIZED COSTS
    model.addConstr(obj["tac"] == sum(c_inv[dev] for dev in all_devs) + sum(c_om[dev] for dev in all_devs) + param["tac_distr"]     # annualized investment costs              
                                  + gas_total * param["price_gas"] + grid_limit_gas * param["price_cap_gas"]                        # gas costs
                                  + from_grid_total * param["price_el"] + grid_limit_el * param["price_cap_el"]                     # electricity costs
 #                                 + (generation_total - to_grid_total) * param["self_charge"]                                       # charge on on-site consumption of CHP-generated power
                                  - to_grid_total * param["revenue_feed_in"]    	                                                 # revenue for grid feed-in
                                  , "sum_up_TAC")                                    
    
    # ANNUAL CO2 EMISSIONS: Implicit emissions by power supply from national grid is penalized, feed-in is ignored
    model.addConstr(obj["co2_gross"] == gas_total * param["gas_CO2_emission"] + from_grid_total * param["grid_CO2_emission"], "sum_up_gross_CO2_emissions")
    
    # POWER PROVIDED BY GRID
    model.addConstr(obj["power_from_grid"] == from_grid_total)
    
    # NET POWER PROVIDED BY GRID
    model.addConstr(obj["net_power_from_grid"] == from_grid_total - to_grid_total)
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Set model parameters and execute calculation
    
    print("Precalculation and model set up done in %f seconds." %(time.time() - start_time))
    
    # Set solver parameters
    model.Params.MIPGap     = param["MIPGap"]             # ---,  gap for branch-and-bound algorithm
    model.Params.method     = 2                           # ---, -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc. (only affects root node)
    model.Params.Heuristics = 0                           # Percentage of time spent on heuristics (0 to 1)
    model.Params.MIPFocus   = 2                           # Can improve calculation time (values: 0 to 3)
    model.Params.Cuts       = 2                           # Cut aggressiveness (values: -1 to 3)
    model.Params.PrePasses  = 8                           # Number of passes performed by presolving (changing can improve presolving time) values: -1 to inf
    
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
        
        # Save results
        save_results(devs, param, dem, model, obj_fn, obj_eps, eps_constr, dir_results)
        
        # Return dictionary
        res_obj = {}        
        for k in set_obj:
            res_obj[k] = obj[k].x
        return res_obj
    
def save_results(devs, param, dem, model, obj_fn, obj_eps, eps_constr, dir_results):
    
    # Convert numpy arrays in parameters to normal lists
    param["T_heating_supply"] = param["T_heating_supply"].tolist()
    param["diameters"]["heating"] = param["diameters"]["heating"].tolist()
    param["diameters"]["cooling"] = param["diameters"]["cooling"].tolist()
    devs["CC"]["COP"] = devs["CC"]["COP"].tolist()
    
    # Write model parameter in json-file
#    all_param = {**param, **devs}
#    with open(dir_results + "\parameter.json", "w") as outfile:
#        json.dump(all_param, outfile, indent=4, sort_keys=True)

    # Write Gurobi files
    model.write(dir_results + "\model.lp")
    model.write(dir_results + "\model.prm")
    model.write(dir_results + "\model.sol")
    
    # Save demands
    with open(dir_results + "\demands.txt", "w") as outfile:
        for com in dem.keys():
            for t in range(8760):
                outfile.write(com + "_t" + str(t) + " " + str(dem[com][t]) + "\n")
                
    # Write further information in txt-file
    with open(dir_results + "\meta_results.txt", "w") as outfile:
        outfile.write("Runtime " + str(round(model.Runtime,6)) + "\n")
        outfile.write("ObjectiveValue " + "{0}".format(model.ObjVal) + "\n")
        outfile.write("ModelStatus " + "{0}".format(model.Status) + "\n")
        outfile.write("NodeCount " + "{0}".format(model.NodeCount) + "\n")
        outfile.write("MIPGap " + "{0}".format(model.Params.MIPGap) + "\n\n")
        outfile.write("ObjectiveFunction " + obj_fn + "\n")
        outfile.write("BoundedFunction " + obj_eps + "\n")
        outfile.write("EpsilonConstraint " + str(eps_constr) + "\n\n")
                    
    print("\nResult files (parameter.json, results.txt, demands.txt, model.lp, model.rpm, model.sol) saved in " + dir_results)