# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 

@author: lkivi
"""


import os

import parameters
import device_optim
import post_processing_clustered as post

import datetime
import numpy as np
import time
import json

import matplotlib.pyplot as plt



# Calculate load curves without phase shift and without cutting  
def calc_dem(peak, total, A, B, C):
                
    time_steps = np.arange(8760)
    dem = {}
    # heating
    dem["heat"] = np.zeros(8760)
    for t in time_steps:
        if t <= T/(2*B["heat"]):
            dem["heat"][t] = A["heat"]*np.cos(B["heat"]*np.pi/T*t)**2 + C["heat"]
        elif t > T/(2*B["heat"]) and t < T - T/(2*B["heat"]):
            dem["heat"][t] = 0
        else:
            dem["heat"][t] = A["heat"]*np.cos(B["heat"]*np.pi/T*(t-T))**2 + C["heat"]    
    # cooling
    dem["cool"] = np.zeros(8760)
    for t in time_steps:
        if t < T/2*(1-1/B["cool"]) or t > T/2*(1-1/B["cool"]) + T/B["cool"]:
            dem["cool"][t] = 0
        else:
            dem["cool"][t] = A["cool"]*np.sin(B["cool"]*np.pi/T*(t- T/2*(1-1/B["cool"])))**2 + C["cool"]
    
    return dem


#%%
# Choose use case
use_case = "DOC_plots"

# Choose scenario
 
scenario = "stand_alone"                     # stand-alone supply
#scenario = "conventional_DHC"                # conventional, separated heating and cooling network
#scenario = "Ectogrid_min"                    # bidirectional network with conventional BU devices and minumum building equipment
#scenario = "Ectogrid_full"                   # bidirectional network with full BU & building equipment



dict_result = {}
dict_result["DOC_dem"] = {}
KPI_list = ["supply_costs", "eta_ex", "FOM_system", "co2_spec", "PE_spec"]
for KPI in KPI_list:
    dict_result[KPI] = {}


    
# number of time steps
T = 8760
time_steps = range(T)

# Peak loads MW
peak = {}
peak["heat"] = 2
peak["cool"] = 2

# yearly demands MWh
# demand has to be lower than peak*8760h !!
# demand wihtout curve compression and minimum value = 0: peak * 4380
total = {}
total["heat"] = 8760
total["cool"] = 8760

# Calculate parameters for load curves
A = {}    # amplitude
B = {}    # compression (>= 1)
C = {}    # minimum value 
for dem in ["heat", "cool"]:
    # If calculated total demand is higher than real total demand: load curve has to be compressed
    if peak[dem] * T/2 > total[dem]:
        A[dem] = peak[dem]
        B[dem] = (peak[dem]*T)/(2*total[dem])
        C[dem] = 0
    # else: minimum value is set > 0
    else:
        A[dem] = 2*(peak[dem] - total[dem]/T)
        B[dem] = 1
        C[dem] = 2*total[dem]/T - peak[dem]   
            
# Calculate DOC without phase shift and without cutting the curves
dem = calc_dem(peak, total, A, B, C)
DOC_ref = 2*np.sum(min(dem["heat"][t], dem["cool"][t]) for t in time_steps) / np.sum((dem["heat"][t] + dem["cool"][t]) for t in time_steps)

# Number of iterations and number of cutting steps
N = 10
N_cut = int(DOC_ref*N)

for item in dict_result:
    dict_result[item] = np.zeros(N+1)

for i in range(N+1):
    
    # Get cutting and phase shifting degree   
    if i == 0:
        cut = 1
        shift = 0
    elif i <= N_cut:
        cut = 1-i/N_cut
        shift = 0
    else:
        cut = 0
        shift = (i-N_cut)/(N-N_cut)
    

    # get original load curves
    dem = calc_dem(peak, total, A, B, C)    
    # Apply cutting
    # heating
    for t in time_steps:
        if t <= T/(2*B["heat"])/2 or t > T - T/(2*B["heat"])/2:
            dem["heat"][t] += cut*(peak["heat"] - dem["heat"][t])
        elif t < T/(2*B["heat"]) or t > T - T/(2*B["heat"]):
            dem["heat"][t] -= cut * dem["heat"][t]
    # cooling
    for t in time_steps:
        if t > T/2*(1-1/(2*B["cool"])) and t < T/2*(1+1/(2*B["cool"])):
            dem["cool"][t] += cut*(peak["cool"] - dem["cool"][t])
        elif t > T/2*(1-1/(1*B["cool"])) or t < T/2*(1+1/(1*B["cool"])):
            dem["cool"][t] -= cut * dem["cool"][t]
                        
    # Apply phase shift to heating demand curve
    dem["heat"] = np.roll(dem["heat"], int(T/2 * shift))
    
    
#    # Plot load curves
#    fig = plt.figure()
#    plt.plot(time_steps, dem["heat"])
#    plt.plot(time_steps, dem["cool"])
    
    
    # Define paths
    path_file = str(os.path.dirname(os.path.realpath(__file__)))
    dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + "_" + scenario
    
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)


    ## Load parameters
    nodes, param, devs, devs_dom = parameters.load_params(use_case, path_file, scenario, dem)
    param["switch_post_processing"] = 0
    
    # Run device optimization
    nodes, param = device_optim.run(nodes, param, devs, devs_dom, dir_results)
    
    # Read solution file
    file_name = dir_results + "\\model.sol"
    with open(file_name, "r") as solution_file:
        file = solution_file.readlines()  
     
    # Load params and node data out of json files    
    param = json.loads(open(dir_results + "\parameter.json" ).read())
    nodes = json.loads(open(dir_results + "\data_nodes.json" ).read())

    # Re-convert lists to arrays
    for item in ["G_sol", "T_cold", "T_hot", "T_soil_deep", "day_matrix", "day_weights", "gas_buildings", "price_el", "sigma", "t_air"]:
            param[item] = np.array(param[item])
    for item in ["CHP", "PV"]:
        param["revenue_feed_in"][item] = np.array(param["revenue_feed_in"][item])                    
    for item in ["T_cooling_return", "T_cooling_supply", "T_heating_return", "T_heating_supply", "cool", "heat", "mass_flow", "power_dem", "res_heat_dem"]:
        for n in nodes:
            nodes[n][item] = np.array(nodes[n][item])
    
    # Calculate KPIs        
    post.calc_KPIs(file, nodes, param, dir_results)        
    KPI_dict = json.loads(open(dir_results + "\System_KPIs.json" ).read())
    
    # Store KPIs of current iteration
    for item in dict_result:
        dict_result[item][i] = KPI_dict[item]
    
    
    
# plot curves
for KPI in KPI_list:    
    fig = plt.figure()   
    plt.plot(dict_result["DOC_dem"], dict_result[KPI])
    plt.xlabel("DOC")
    plt.ylabel(KPI)
    
    
    
