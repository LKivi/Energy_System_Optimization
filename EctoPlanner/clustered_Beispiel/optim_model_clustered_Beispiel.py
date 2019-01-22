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
import matplotlib.pyplot as plt
import pandas as pd

def run_optim(obj_fn, obj_eps, eps_constr, dir_results, clustered, part_load):
    assert (obj_eps == "" and eps_constr == "") or (obj_eps != "" and eps_constr != ""), "If there is a bounded objective function, an epsilon constraint should be given."
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load model parameter
    start_time = time.time()
    
    days = range(clustered["days"])
    time_steps = range(clustered["time_steps"])
    
    (devs, param, dem) = parameter.load_params(clustered)

    # Create set for devices
    all_devs = ["BOI", "CHP", "AC", "CC", "TES"]       
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = gp.Model("Basic_Model")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables

    # Purchase decision binary variables (1 if device is installed, 0 otherwise)
    x = {}
    for device in all_devs:
        x[device] = model.addVar(vtype="B", name="x_" + str(device))
    
    if part_load==1:
        y = {}  # Binary. 1 if device is activated, 0 otherwise
        for d in days:
            for t in time_steps:
                for device in ["BOI", "CHP","AC"]:
                    y[device,d,t] = model.addVar(vtype="B", name="activation_" + str(device) + "_d" + str(d) + "_t" + str(t))  
                
        heat_nom = {}  # Nominal heat of Boiler
        for d in days:
            for t in time_steps:
                for device in ["BOI"]:
                    heat_nom[device,d,t] = model.addVar(vtype="C", name="nominal_heat_" + str(device) + "_d" + str(d) + "_t" + str(t))   

        cool_nom = {}  # Nominal cool of AC
        for d in days:
            for t in time_steps:
                for device in ["AC"]:
                    cool_nom[device,d,t] = model.addVar(vtype="C", name="nominal_cool_" + str(device) + "_d" + str(d) + "_t" + str(t))    
                    
                    
        power_nom = {}  # Nominal power of CHP
        for d in days:
            for t in time_steps:
                for device in ["CHP"]:
                    power_nom[device,d,t] = model.addVar(vtype="C", name="nominal_power_" + str(device) + "_d" + str(d) + "_t" + str(t))   
                    
            
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["BOI", "CHP", "AC", "CC", "TES"]:
        cap[device] = model.addVar(vtype="C", name="nominal_capacity_" + str(device))
        
    #%%  Energy transfers
    
    power = {}
    heat = {}
    cool = {}
    gas = {}
    
    for d in days:
        for t in time_steps:        
            timetag = "_d"+str(d)+"_t"+str(t)
            
            # Gas to/from devices
            for device in ["BOI", "CHP"]:
                gas[device,d,t] = model.addVar(vtype="C", name="gas_" + device + timetag)
                
            # Eletrical Power to/from devices    
            for device in ["CHP", "CC", "from_grid", "to_grid"]:
                power[device,d,t] = model.addVar(vtype="C", name="power_" + device + timetag)
            
            # Heat to/from devices
            for device in ["BOI", "CHP", "AC"]:
                heat[device,d,t] = model.addVar(vtype="C", name="heat_" + device + timetag)
            
            # Cooling power to/from devices
            for device in ["CC", "AC"]:
                cool[device,d,t] = model.addVar(vtype="C", name="cool_" + device + timetag)
    
    #%% Storage variables
    ch = {}  # Energy flow to charge storage device
    dch = {} # Energy flow to discharge storage device
    soc = {} # State of charge
    soc_init = {} # Initial State of charge of every day
    
    for device in ["TES"]:
        for d in days:        
            soc_init[device,d] = model.addVar(vtype="C", name="SOC_init_" + device + "_" + str(d))
            
            for t in time_steps:        
                timetag = "_"+str(d)+"_"+str(t)           
                ch[device,d,t] = model.addVar(vtype="C", name="ch_" + device + "_d" + str(d) + "_t" + str(t))
                dch[device,d,t] = model.addVar(vtype="C", name="dch_" + device + "_d" + str(d) + "_t" + str(t))
                soc[device,d,t] = model.addVar(vtype="C", name="soc_" + device + "_d" + str(d) + "_t" + str(t))
        
    # Objective functions
    obj = {}
    set_obj = ["tac", "co2_gross", "power_from_grid", "net_power_from_grid"]
    for k in set_obj:
        obj[k] = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="obj_" + k)    
      
   #%% Define objective function

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


  #%% Add constraints
    
    #%% CONTINUOUS SIZING OF DEVICES: minimum capacity <= capacity <= maximum capacity
    for device in ["TES"]:
        model.addConstr(cap[device] <= x[device] * devs[device]["max_cap"])
        model.addConstr(cap[device] >= x[device] * devs[device]["min_cap"])
    
    for d in days:
        for t in time_steps:
            for device in ["BOI"]:
                model.addConstr(heat[device,d,t] <= cap[device])
                
#            for device in ["CHP"]:
#                model.addConstr(power[device,d,t] <= cap[device])
            
            for device in ["CC"]:
                model.addConstr(cool[device,d,t] <= cap[device])
                
#%% Part load constraints
    
    if part_load == 1:
        # Constraints for CHP
        for d in days:
            for t in time_steps:
                for device in ["CHP"]:
                    model.addConstr(y[device,d,t] <= x[device])
    
        for device in ["CHP"]:
            model.addConstr(cap[device] >= x[device] * devs[device]["min_cap"])
            model.addConstr(cap[device] <= x[device] * devs[device]["max_cap"])
        
        for d in days:
            for t in time_steps:
                for device in ["CHP"]:
                    model.addConstr(power_nom[device,d,t] >= y[device,d,t] * devs[device]["min_cap"])
                    model.addConstr(power_nom[device,d,t] <= y[device,d,t] * devs[device]["max_cap"])
                
                    model.addConstr((cap[device]-power_nom[device,d,t]) >= (x[device]-y[device,d,t]) * devs[device]["min_cap"])
                    model.addConstr((cap[device]-power_nom[device,d,t]) <= (x[device]-y[device,d,t]) * devs[device]["max_cap"])
                    
                    model.addConstr(power[device,d,t] >= devs[device]["min_load"] * power_nom[device,d,t])
                    model.addConstr(power[device,d,t] <= power_nom[device,d,t])
                    
        # Constraints for AC            
        for d in days:
            for t in time_steps:
                for device in ["AC"]:
                    model.addConstr(y[device,d,t] <= x[device])
    
        for device in ["AC"]:
            model.addConstr(cap[device] >= x[device] * devs[device]["min_cap"])
            model.addConstr(cap[device] <= x[device] * devs[device]["max_cap"])
        
        for d in days:
            for t in time_steps:
                for device in ["AC"]:
                    model.addConstr(cool_nom[device,d,t] >= y[device,d,t] * devs[device]["min_cap"])
                    model.addConstr(cool_nom[device,d,t] <= y[device,d,t] * devs[device]["max_cap"])
                
                    model.addConstr((cap[device]-cool_nom[device,d,t]) >= (x[device]-y[device,d,t]) * devs[device]["min_cap"])
                    model.addConstr((cap[device]-cool_nom[device,d,t]) <= (x[device]-y[device,d,t]) * devs[device]["max_cap"])
                    
                    model.addConstr(cool[device,d,t] >= devs[device]["min_load"] * cool_nom[device,d,t])
                    model.addConstr(cool[device,d,t] <= cool_nom[device,d,t])
                    
                    
# Constraints if partload behaviour is not considered                    
    else:                    
        # Constraint for CHP            
        for d in days:
            for t in time_steps:
                for device in ["CHP"]:
                    model.addConstr(power[device,d,t] <= cap[device])
                    
        # Constraint for AC            
        for d in days:
            for t in time_steps:
                for device in ["AC"]:
                    model.addConstr(cool[device,d,t] <= cap[device])
                              
                
    

    #%% INPUT / OUTPUT CONSTRAINTS
    for d in days:
        for t in time_steps:
            # Boiler
            gas["BOI",d,t] = heat["BOI",d,t] / devs["BOI"]["eta_th"]
            
            # Combined heat and power
            model.addConstr(power["CHP",d,t] == heat["CHP",d,t] / devs["CHP"]["eta_th"] * devs["CHP"]["eta_el"])
            model.addConstr(gas["CHP",d,t] == heat["CHP",d,t] / devs["CHP"]["eta_th"])
            
            # Compression chiller
            model.addConstr(cool["CC",d,t] == power["CC",d,t] * devs["CC"]["COP"])  

            
            # Absorption chiller
            model.addConstr(cool["AC",d,t] == heat["AC",d,t] * devs["AC"]["eta_th"])

    #%% ENERGY BALANCES
    for d in days:
        for t in time_steps:
            # Heat balance
            model.addConstr(heat["BOI",d,t] + heat["CHP",d,t] + dch["TES",d,t] == dem["heat"][d,t] + heat["AC",d,t] + ch["TES",d,t])

        for t in time_steps:
            # Electricity balance
            model.addConstr(power["CHP",d,t] + power["from_grid",d,t] == dem["power"][d,t] + power["to_grid",d,t] + power["CC",d,t])

        for t in time_steps:
            # Cooling balance
            model.addConstr(cool["AC",d,t] + cool["CC",d,t] == dem["cool"][d,t])    
    
    #%% STORAGE DEVICES
    for device in ["TES"]:  
        for d in days:
            model.addConstr(cap[device] >= soc_init[device,d], name="cap_inits_"+device+"_"+str(d))
            for t in time_steps:
                model.addConstr(cap[device] >= soc[device,d,t], name="cap_" +device+"_"+str(d)+"_"+str(t))
                
        for d in range(clustered["days"]):
            if np.max(clustered["weights"]) > 1:
                model.addConstr(soc_init[device,d] == soc[device,d,clustered["time_steps"]-1], name="repetitions_" +device+"_"+str(d))
                              
        k_loss = devs[device]["sto_loss"]
        eta_ch  = devs[device]["eta_ch"]
        eta_dch = devs[device]["eta_dch"]
        
        for d in days:
            for t in time_steps:
                if t == 0:
                    #if np.max(clustered["weights"]) == 1:
                    #    if d == 0:
                    #       soc_prev = soc_init[device,d]
                    #    else:
                    #       soc_prev = soc[device,d-1,clustered["time_steps"]-1]
                    #else:
                        soc_prev = soc_init[device,d]
                        #vielleicht: soc[device,d,t] = soc_init[device,d]
                else:
                    soc_prev = soc[device,d,t-1]
                    # soc_prev weglassen?
                
                timetag = "_" + str(d) + "_" + str(t)
                
                charge = eta_ch * ch[device,d,t]
                discharge = 1 / eta_dch * dch[device,d,t]
                
                model.addConstr(soc[device,d,t] == (1 - k_loss) * soc_prev + (charge - discharge),
                                name="Storage_bal_"+device+timetag)

    #%% SUM UP RESULTS
    gas_total = sum(sum(clustered["weights"][d] * sum(gas[device,d,t] 
                for t in time_steps) for d in days)       
                for device in ["BOI", "CHP"])
  
    from_grid_total = sum(sum(clustered["weights"][d] * power["from_grid",d,t] 
                      for t in time_steps) for d in days)
    to_grid_total = sum(sum(clustered["weights"][d] * power["to_grid",d,t] 
                    for t in time_steps)for d in days)
#%% Econnomic Constraints
    
    # Investments
    c_inv = {}
    for device in all_devs:
        c_inv[device] = cap[device] * devs[device]["ann_inv_var"]

    # Operation and maintenance costs
    c_om = {}
    for device in all_devs: 
        c_om[device] = devs[device]["cost_om"] * (cap[device] * devs[device]["inv_var"])

    #%% OBJECTIVE FUNCTIONS
    # TOTAL ANNUALIZED COSTS
    model.addConstr(obj["tac"] == sum(c_inv[dev] for dev in all_devs) + sum(c_om[dev] for dev in all_devs)  
                                  + gas_total * param["price_gas"] + from_grid_total * param["price_el"] - to_grid_total * param["revenue_feed_in"], "sum_up_TAC")
    
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
    model.Params.MIPGap     = param["MIPGap"]   # ---,         gap for branch-and-bound algorithm
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
        # Save results
        save_results(devs, param, dem, model, obj_fn, obj_eps, eps_constr, dir_results, clustered)
        
        # Return dictionary
        res_obj = {}   
        res_cap = {}
        for k in set_obj:
            res_obj[k] = obj[k].x
            
        for k in ["BOI", "CHP", "AC", "CC", "TES"]:
            res_cap[k] = cap[k].x
            
        return {**res_obj, **res_cap}
    
def save_results(devs, param, dem, model, obj_fn, obj_eps, eps_constr, dir_results, clustered):
    
    days = range(clustered["days"])
    time_steps = range(clustered["time_steps"])
    
    # Write model parameter in json-file
    all_param = {**param, **devs}
    with open(dir_results + "\parameter.json", "w") as outfile:
        json.dump(all_param, outfile, indent=4, sort_keys=True)

    # Write Gurobi files
    model.write(dir_results + "\model.lp")
    model.write(dir_results + "\model.prm")
    model.write(dir_results + "\model.sol")
    
    # Save demands
    with open(dir_results + "\demands.txt", "w") as outfile:
        for com in dem.keys():
            for d in days:
                for t in time_steps:
                    outfile.write(com + "_d" + str(d) + "_t" + str(t) + " " + str(dem[com][d,t]) + "\n")
                
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
    
    
    # Create LP Matrix Plot
    nzs = pd.DataFrame(get_matrix_coos(model), columns=['row_idx', 'col_idx', 'coeff'])
    plt.scatter(nzs.col_idx, nzs.row_idx, marker='.', lw=0)
    plt.savefig(dir_results + "\LP_Matrix")
    
def get_expr_coos(expr, var_indices):
    for i in range(expr.size()):
        dvar = expr.getVar(i)
        yield expr.getCoeff(i), var_indices[dvar]
        
def get_matrix_coos(model):
    dvars = model.getVars()
    constrs = model.getConstrs()
    var_indices = {v: i for i, v in enumerate(dvars)}
    for row_idx, constr in enumerate(constrs):
        for coeff, col_idx in get_expr_coos(model.getRow(constr), var_indices):
            yield row_idx, col_idx, coeff
    
    