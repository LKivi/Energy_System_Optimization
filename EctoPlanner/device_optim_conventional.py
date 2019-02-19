# -*- coding: utf-8 -*-
"""

Author: Marco Wirtz, Institute for Energy Efficient Buildings and Indoor Climate, RWTH Aachen University, Germany

Created: 01.09.2018

"""

from __future__ import division
import gurobipy as gp
import os
import json
import time
import numpy as np
import post_processing_clustered as post



def run_optim(obj_fn, obj_eps, eps_constr, nodes, param, devs, devs_dom, dir_results):

    assert (obj_eps == "" and eps_constr == "") or (obj_eps != "" and eps_constr != ""), "If there is a bounded objective function, an epsilon constraint should be given."
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load model parameter
    start_time = time.time()   
          
    time_steps = range(8760)
    
    # Sum up building heat demands
    dem = {}
    for demand in ["heat", "cool"]:
        dem[demand] = sum(nodes[n][demand] for n in nodes) / 1000
        

    # Create set for devices
    all_devs = ["BOI", "CHP", "AC", "CC", "SUB"]         # SUB = building substations ( = heat exchangers)
    
    
    # Get constant investment costs (kEUR / MW)
    inv_var = {}
    inv_var["BOI"] = 67.5
    inv_var["CHP"] = 768
    inv_var["AC"] = 525
    inv_var["CC"] = 166  
    
    
    # Substation (= heat exchangers) parameters equal free-cooler parameters
    devs["SUB"] = devs_dom["FRC"]
    inv_var["SUB"] = devs["SUB"]["inv_var"]
    devs["SUB"]["ann_factor"] = devs["SUB"]["ann_inv_var"] / devs["SUB"]["inv_var"]
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("DHC_Benchmark")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables

    
    # Piece-wise linear function variables
    lin = {}
    for device in ["BOI", "CHP", "AC", "CC"]:   
        lin[device] = {}
        for i in range(len(devs[device]["cap_i"])):
            lin[device][i] = model.addVar(vtype="C", name="lin_" + device + "_i" + str(i))
            
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["BOI", "CHP", "AC", "CC", "SUB"]:
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
    for device in ["BOI", "CHP", "AC"]:
        heat[device] = {}
        for t in time_steps:
            heat[device][t] = model.addVar(vtype="C", name="heat_" + device + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "AC"]:
        cool[device] = {}
        for t in time_steps:
            cool[device][t] = model.addVar(vtype="C", name="cool_" + device + "_t" + str(t))
            
    inv = {}
    c_inv = {}
    c_om = {}
    c_total = {}
    for device in all_devs:
        inv[device] = model.addVar(vtype = "C", name="investment_costs_" + device)
    for device in all_devs:
        c_inv[device] = model.addVar(vtype = "C", name="annual_investment_costs_" + device)
    for device in all_devs:
        c_om[device] = model.addVar(vtype = "C", name="om_costs_" + device)
    for device in all_devs:
        c_total[device] = model.addVar(vtype = "C", name="total_annual_costs_" + device)        
    

  
    # grid maximum transmission power
    grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
    grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")
    
    # total energy amounts taken from grid and fed into grid
    from_grid_total = model.addVar(vtype = "C", name="from_grid_total")
    to_grid_total = model.addVar(vtype = "C", name="to_grid_total")
    gas_total = model.addVar(vtype = "C", name="gas_total")
    
    # total revenue for feed-in
    revenue_feed_in = model.addVar(vtype="C", name="revenue_feed_in_CHP")
    # Electricity costs
    electricity_costs = model.addVar(vtype = "C", name="electricity_costs")    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Objective functions
    obj = {}
    set_obj = ["tac", "co2_gross"] # Mögliche Zielgrößen
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
    
    if param["switch_cost_functions"]:
    
        for device in ["BOI", "CHP", "AC", "CC"]:
        
            model.addConstr(cap[device] == sum(lin[device][i] * devs[device]["cap_i"][i] for i in range(len(devs[device]["cap_i"]))))
            # lin: Special Ordered Sets of type 2 (SOS2 or S2): an ordered set of non-negative variables, of which at most two can be non-zero, and if 
            # two are non-zero these must be consecutive in their ordering. 
            model.addSOS(gp.GRB.SOS_TYPE2, [lin[device][i] for i in range(len(devs[device]["cap_i"]))])
            
            # Sum of linear function variables should be 1
            model.addConstr(1 == sum(lin[device][i] for i in range(len(devs[device]["cap_i"]))))
            
            

      

    
    #%% LOAD CONTRAINTS: minimal load < load < capacity
    

    for t in time_steps:
        for device in ["BOI"]:
            model.addConstr(heat[device][t] <= cap[device])
            
        for device in ["CHP"]:
            model.addConstr(power[device][t] <= cap[device])
        
        for device in ["CC", "AC"]:
            model.addConstr(cool[device][t] <= cap[device])
                  
        # limitation of power from and to grid   
        model.addConstr(sum(gas[device][t] for device in ["BOI", "CHP"]) <= grid_limit_gas)       
        for device in ["from_grid", "to_grid"]:
            model.addConstr(power[device][t] <= grid_limit_el)
            
        
    # Sum of substation capacities
    model.addConstr( cap["SUB"] == sum(sum( np.max(nodes[n][demand]) for demand in ["heat", "cool"]) for n in nodes) / 1000 )

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
        


    #%% GLOBAL ENERGY BALANCES

    for t in time_steps:
        # Heat balance
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] == dem["heat"][t] + heat["AC"][t])

        # Electricity balance
        model.addConstr(power["CHP"][t] + power["from_grid"][t] == power["to_grid"][t] + power["CC"][t])

        # Cooling balance
        model.addConstr(cool["AC"][t] + cool["CC"][t] == dem["cool"][t])
        
    
    # Absorption chiller can only be supplied by Boiler and CHP
    for t in time_steps:
        model.addConstr(heat["BOI"][t] + heat["CHP"][t] >= heat["AC"][t])    
        
    
            
        
    #%% SUM UP RESULTS
    model.addConstr(gas_total == sum(sum(gas[device][t] for t in time_steps) for device in ["BOI", "CHP"]))
  
    model.addConstr(from_grid_total == sum(power["from_grid"][t] for t in time_steps) )
    model.addConstr(to_grid_total == sum(power["to_grid"][t] for t in time_steps) )

    model.addConstr(electricity_costs == sum((power["from_grid"][t] * param["price_el"][t]) for t in time_steps) )
    
    model.addConstr(revenue_feed_in == sum((power["to_grid"][t] * param["revenue_feed_in"]["CHP"][t]) for t in time_steps) )
    
    
    
    # total investment costs
    if param["switch_cost_functions"]:
        for device in ["BOI", "CHP", "CC", "AC"]:
            model.addConstr( inv[device] == sum(lin[device][i] * devs[device]["inv_i"][i] for i in range(len(devs[device]["cap_i"]))) )
        for device in ["SUB"]:
            model.addConstr(inv[device] == inv_var[device] * cap[device])
    else:
        for device in all_devs:
            model.addConstr(inv[device] == inv_var[device] * cap[device])        
        
    # Annual investment costs
    for device in all_devs:
        model.addConstr( c_inv[device] == inv[device] * devs[device]["ann_factor"] )
    
    # Operation and maintenance costs
    for device in all_devs:       
        model.addConstr( c_om[device] == devs[device]["cost_om"] * inv[device] )
    
    # Total annual costs
    for device in all_devs:
        model.addConstr( c_total[device] == c_inv[device] + c_om[device] )
        

    
    

    #%% OBJECTIVE FUNCTIONS
    # TOTAL ANNUALIZED COSTS
    model.addConstr(obj["tac"] == sum(c_total[dev] for dev in all_devs)                                           # annualized investment costs              
                                  + gas_total * param["price_gas"] + grid_limit_gas * param["price_cap_gas"]      # gas costs
                                  + electricity_costs + grid_limit_el * param["price_cap_el"]                     # electricity costs
                                  - revenue_feed_in   	                                                          # revenue for grid feed-in
                                  , "sum_up_TAC")                                    
    
    # ANNUAL CO2 EMISSIONS: Implicit emissions by power supply from national grid is penalized, feed-in is ignored
    model.addConstr(obj["co2_gross"] == gas_total * param["gas_CO2_emission"] + from_grid_total * param["grid_CO2_emission"], "sum_up_gross_CO2_emissions")

    
    
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
        
        # Write Gurobi files
        model.write(dir_results + "\model.lp")
        model.write(dir_results + "\model.prm")
        model.write(dir_results + "\model.sol")


        # Print tac
        print("tac: " + str(obj[obj_fn].X))
        

        # Store param and nodes as json

        # param
        for item in ["G_sol", "T_cold", "T_hot", "T_soil_deep", "price_el", "t_air"]:
            param[item] = param[item].tolist()
        for item in ["CHP", "PV"]:
            param["revenue_feed_in"][item] = param["revenue_feed_in"][item].tolist()                
        with open(dir_results + "\parameter.json", "w") as outfile:
            json.dump(param, outfile, indent=4, sort_keys=True)    
        
        # nodes
        for item in ["T_cooling_return", "T_cooling_supply", "T_heating_return", "T_heating_supply", "cool", "heat"]:
            for n in nodes:
                nodes[n][item] = nodes[n][item].tolist()
        with open(dir_results + "\data_nodes.json", "w") as outfile:
            json.dump(nodes, outfile, indent=4, sort_keys=True)         


        # Run Post Processing
#        if param["switch_post_processing"]:
#            post.run(dir_results)
    
        
        # Print real investment costs per MW
        # ( To Check if input costs (line 48 ff.) are reasonable )
        for device in ["BOI", "CHP", "AC", "CC"]:
            if cap[device].X > 0:
                # Calculate investment costs according to piece-wise linear funcitons
                for i in range(len(devs[device]["cap_i"])):
                    if devs[device]["cap_i"][i] >= cap[device].X:
                        # Get supporting points for linear interpolation
                        cap_top = devs[device]["cap_i"][i]
                        cap_bot = devs[device]["cap_i"][i-1]
                        inv_top = devs[device]["inv_i"][i]
                        inv_bot = devs[device]["inv_i"][i-1]
                        break
                # Calculate real variable investment
                inv_var_real = (inv_bot + (cap[device].X - cap_bot) / (cap_top - cap_bot) * (inv_top - inv_bot)) / cap[device].X                       
                print(device + ": " + str(round(inv_var_real,2)))
                
        
    
        # return nodes, param
        return nodes, param













