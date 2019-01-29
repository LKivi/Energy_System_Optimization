# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:45:22 2019

@author: lkivi
"""

import parameters
import gurobipy as gp
import time
import os


#%% STAND-ALONE OPTIMIZATION

use_case = "FZJ"
scenario = "standalone_typedays"
path_file = str(os.path.dirname(os.path.realpath(__file__))) 
dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + "_" + scenario

# Load Params
nodes, param, devs, devs_dom = parameters.load_params(use_case, path_file, scenario)

# Measure calculation time
start_time = time.time()

# Steps
time_steps = range(24)
n_days = param["n_clusters"]
days= range(n_days)

# List of all building devices
all_devs_dom = ["HP", "CC", "BOI"] 

# Create a new model
model = gp.Model("Stand_Alone")

#%% VARIABLES

# Device's capacity (i.e. nominal power)
cap_dom = {}
for device in ["HP", "CC", "BOI"]:
    cap_dom[device] = {}
    for n in nodes:
        cap_dom[device][n] = model.addVar(vtype="C", name="nominal_capacity_" + str(device) + "_n" + str(n))
        
# Eletrical power to/from devices
power_dom = {}
for device in ["HP", "CC", "from_grid"]:
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
for device in ["HP", "BOI"]:
   heat_dom[device] = {}
   for n in nodes:
       heat_dom[device][n] = {}  
       for d in days:
           heat_dom[device][n][d] = {}
           for t in time_steps:
               heat_dom[device][n][d][t] = model.addVar(vtype="C", name="heat_"  + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))

# Cooling power to/from devices
cool_dom = {}
for device in ["CC"]:
    cool_dom[device] = {}
    for n in nodes:
        cool_dom[device][n] = {}
        for d in days:
            cool_dom[device][n][d] = {}
            for t in time_steps:
                cool_dom[device][n][d][t] = model.addVar(vtype="C", name="cool_" + device + "_n" + str(n) + "_d" + str(d) + "_t" + str(t))
                
                
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
        c_total_dom[device][n]  = model.addVar(vtype="C", name="total_annual_costs_" + device + "_n" + str(n))    
        

# grid maximum transmission power
grid_limit_el = model.addVar(vtype = "C", name="grid_limit_el")  
grid_limit_gas = model.addVar(vtype = "C", name="grid_limit_gas")

# total energy amounts taken from grid
from_grid_total = model.addVar(vtype = "C", name="from_grid_total")
# total power to grid
to_grid_total = model.addVar(vtype = "C", name="to_grid_total")
# Gas taken from grid
gas_total = model.addVar(vtype = "C", name="gas_total")
# Electricity costs
electricity_costs = model.addVar(vtype = "C", name="electricity_costs")
    
# Total annualized costs
tac_total = model.addVar(vtype = "c", name = "total_annualized_costs")          
            
# Objective functions
obj = model.addVar(vtype="C", lb=-gp.GRB.INFINITY, name="obj")                      
                
                
#%% Define objective function
model.update()
model.setObjective(obj, gp.GRB.MINIMIZE)




#%% LOAD CONSTRAINTS
for n in nodes:
    for d in days:
        for t in time_steps:              
            
            for device in ["HP", "BOI"]:
                model.addConstr(heat_dom[device][n][d][t] <= cap_dom[device][n])
            
            for device in ["CC"]:
                model.addConstr(cool_dom[device][n][d][t] <= cap_dom[device][n])


for n in nodes:
    model.addConstr(cap_dom["HP"][n] <= 100)

 #%% INPUT / OUTPUT CONSTRAINTS

for n in nodes:
    for d in days:
        for t in time_steps:
            
            # Compression chiller
            model.addConstr(cool_dom["CC"][n][d][t] == power_dom["CC"][n][d][t] * devs_dom["CC"]["COP"][n][d][t])  
    
            # Heat Pump
            model.addConstr(heat_dom["HP"][n][d][t] == power_dom["HP"][n][d][t] * devs_dom["HP"]["COP"][n][d][t])
            
            # Boiler
            model.addConstr(heat_dom["BOI"][n][d][t] == gas_dom["BOI"][n][d][t] * devs_dom["BOI"]["eta_th"])
                                   
                
#%% ENERGY BALANCES
        
for n in nodes:
    for d in days:
        for t in time_steps:
            
            # Heat balance
            model.addConstr(heat_dom["HP"][n][d][t] + heat_dom["BOI"][n][d][t] == nodes[n]["heat"][d][t] )  
    
            # Cooling balance
            model.addConstr(cool_dom["CC"][n][d][t] == nodes[n]["cool"][d][t] ) 
            
            # Electricity balance
            model.addConstr(power_dom["from_grid"][n][d][t] == power_dom["HP"][n][d][t] + power_dom["CC"][n][d][t])
                                             
                

#%% INVESTMENT COSTS
 
for device in all_devs_dom:

    for n in nodes:
        
        # investment costs
        model.addConstr( inv_dom[device][n] == devs_dom[device]["inv_var"] * cap_dom[device][n] )
    
        # annualized investment
        model.addConstr( c_inv_dom[device][n] == devs_dom[device]["ann_inv_var"] * cap_dom[device][n] + devs_dom[device]["ann_inv_fix"] )

        # Operation and maintenance costs 
        model.addConstr( c_om_dom[device][n] == devs_dom[device]["cost_om"] * inv_dom[device][n] )
        
        # Tac for building device (kEUR)
        model.addConstr( c_total_dom[device][n] == (c_inv_dom[device][n] + c_om_dom[device][n]) / 1000)                    



#%% GRID LIMITS
                
# limitation of power from and to grid  
for d in days:
    for t in time_steps:
        model.addConstr(sum(gas_dom["BOI"][n][d][t] for n in nodes) / 1000 <= grid_limit_gas)       
        model.addConstr(sum(power_dom["from_grid"][n][d][t] for n in nodes) / 1000 <= grid_limit_el) 
        
        

#%% SUM UP

model.addConstr(gas_total == sum(sum(sum((gas_dom["BOI"][n][d][t] * param["day_weights"][d]) for t in time_steps) for d in days) for n in nodes) / 1000)
  
model.addConstr(from_grid_total == sum(sum(sum((power_dom["from_grid"][n][d][t] * param["day_weights"][d]) for t in time_steps) for d in days) for n in nodes) / 1000)
    
model.addConstr(electricity_costs == sum(sum(sum((power_dom["from_grid"][n][d][t] * param["price_el"][d][t] * param["day_weights"][d]) for t in time_steps) for d in days) for n in nodes) / 1000)                
                


#%% OBJECTIVE

model.addConstr(tac_total ==      sum(sum(c_total_dom[dev][n] for n in nodes) for dev in all_devs_dom)                # annual investment costs + o&m costs for building devices
                                + gas_total * param["price_gas"] + grid_limit_gas * param["price_cap_gas"]            # gas costs
                                + electricity_costs + grid_limit_el * param["price_cap_el"]                           # electricity purchase costs + capacity price for grid usage
                                , "sum_up_TAC")

model.addConstr(obj == tac_total)



               
                
#%% Set model parameters and execute calculation
    
print("Precalculation and model set up done in %f seconds." %(time.time() - start_time))

# Set solver parameters
model.Params.MIPGap     = param["MIPGap"]           # ---,         gap for branch-and-bound algorithm
model.Params.method     = 2                         # ---,         -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc.
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

# Create result directory
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

# Write Gurobi files
model.write(dir_results + "\model.lp")
model.write(dir_results + "\model.prm")
model.write(dir_results + "\model.sol")
        
# Print tac
print("tac: " + str(obj.X))                
                
                
                
                
                
                
                
                
                
                
                
                
                
                