# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19

@author: lki
"""


import matplotlib.pyplot as plt
import os
import numpy as np
import json
#import time


#%% RUN POST-PROCESSING
def run(dir_results):
    

    # Read solution file
    file_name = dir_results + "\\model.sol"
    with open(file_name, "r") as solution_file:
        file = solution_file.readlines()

    # Create folder for plots
    dir_plots = dir_results + "\\Plots"
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)
     
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
    

    # Create plots

#    # Building balance plots
#    plot_bldg_balancing(file, param, nodes, dir_plots + "\\building_balances")
##    
#    # BU balance plots
#    plot_BU_balances(file, param, nodes, dir_plots + "\\BU_balances")
        
#    # load plots
#    plot_load_charts(file, param, nodes, dir_plots + "\\load_charts")
        
    # Plot capacities and generation
    plot_capacities(file, param, nodes, dir_plots + "\\capacity_plots")
    
#    # Plot BU storage SOCs
#    plot_storage_soc(file, param, dir_plots + "\\soc_storages")
    
    # Plot building sum
#    plot_flexibility(file, param, nodes, dir_plots + "\\flexibility")
    
#    # Plot sorted annual load curves
#    plot_sorted_curves(file, param, dir_plots + "\\sorted_load_curves")

    # Plot cost structure
    plot_costs(file, param, nodes, dir_plots)
    
#    # Plot balancing matrix
#    plot_balancing_matrix(file, param, nodes, dir_plots)
    
    # Calculate system KPIs
    calc_KPIs(file, nodes, param, dir_results)
    
    


#%%
def calc_KPIs(file, nodes, param, dir_results):
 
    
    print("Calculating system KPIs...")     
    
    time_steps = range(24)
    n_days = param["n_clusters"]
    
    dict_KPI = {}
    
    
    # Cost KPIs
    # Costs for thermal energy supply (EUR/MWh)
    # Read total annualized costs (EUR)
    for line in range(len(file)):
        if "total_annualized_costs" in file[line]:
            dict_KPI["tac_total"] = float(str.split(file[line])[1])
            break            
    # total thermal energy demand (heating and cooling) (MWh)
    dem_total = sum(sum(sum(sum((nodes[n][dem][d][t] * param["day_weights"][d]) for dem in ["heat", "cool"]) for t in time_steps) for d in range(n_days)) for n in nodes) / 1000    
    # Calculate supply costs EUR/MWh
    dict_KPI["supply_costs"] = dict_KPI["tac_total"] * 1000 / dem_total
    
    
    
    # Overlap coefficients
    # Demands
    dem = {}
    dem["heat"] = {}
    dem["cool"] = {}
    for d  in range(n_days):
        dem["heat"][d] = {}
        dem["cool"][d] = {}
        for t in time_steps:
            dem["heat"][d][t] = sum(nodes[n]["heat"][d][t] for n in nodes)
            dem["cool"][d][t] = sum(nodes[n]["cool"][d][t] for n in nodes)
    dict_KPI["DOC_dem"] = ( 2 * sum(sum( min( dem["heat"][d][t], dem["cool"][d][t] ) * param["day_weights"][d] for d in range(n_days)) for t in time_steps)) / (dem_total*1000)
    
    
    # Internal
    dict_KPI["DOC_BES"] = {}
    if param["switch_bidirectional"]:
        # collect relevant flows
        flows = ["heat_HP", "power_HP",
                 "cool_CC", "power_CC",
                 "cool_FRC"]
        BES_heating = {}
        BES_cooling = {}
        series = {}
        for n in nodes:
            series[n] = {}
            for flow in flows:
                series[n][flow] = read_energy_flow(file, flow, n, param)
            BES_heating[n] = series[n]["heat_HP"] - series[n]["power_HP"]
            BES_cooling[n] = series[n]["cool_CC"] + series[n]["power_CC"] + series[n]["cool_FRC"]
            
        for n in nodes:
            dict_KPI["DOC_BES"][n] = ( 2 * sum(sum( min(BES_heating[n][d][t], BES_cooling[n][d][t]) * param["day_weights"][d] for d in range(n_days)) for t in time_steps)) / ( sum(sum((BES_heating[n][d][t] + BES_cooling[n][d][t]) * param["day_weights"][d] for d in range(n_days)) for t in time_steps))
    
            
        dict_KPI["DOC_BES"]["sum"] = ( 2 * sum(sum(sum( min(BES_heating[n][d][t], BES_cooling[n][d][t]) * param["day_weights"][d] for d in range(n_days)) for t in time_steps) for n in nodes)) / ( sum(sum(sum((BES_heating[n][d][t] + BES_cooling[n][d][t]) * param["day_weights"][d] for d in range(n_days)) for t in time_steps) for n in nodes))
    else:
        dict_KPI["DOC_BES"]["sum"] = 0
    
    
    # Network
    if param["switch_bidirectional"]:
        # Sum up building residual loads
        network_heating_demand = {}
        network_cooling_demand = {}
        for n in nodes:
            network_heating_demand[n] = np.zeros((n_days, len(time_steps)))
            network_cooling_demand[n] = np.zeros((n_days, len(time_steps)))
            res_thermal = read_energy_flow(file, "residual_thermal", n, param)
            for d in range(n_days):
                for t in time_steps:
                    if res_thermal[d][t] > 0:
                        network_heating_demand[n][d][t] += res_thermal[d][t]
                    else:
                        network_cooling_demand[n][d][t] += -res_thermal[d][t] 
        dict_KPI["DOC_N"] = ( 2 * sum(sum( min( sum( network_heating_demand[n][d][t] for n in nodes), sum(network_cooling_demand[n][d][t] for n in nodes)) * param["day_weights"][d] for d in range(n_days)) for t in time_steps)) / (sum(sum(sum((network_heating_demand[n][d][t] + network_cooling_demand[n][d][t]) * param["day_weights"][d] for n in nodes) for d in range(n_days)) for t in time_steps))                      
    else:
        dict_KPI["DOC_N"] = 0
    
    
    # Exergy efficiency
    dem_heat = sum(sum(sum(nodes[n]["heat"][d][t] * param["day_weights"][d] for d in range(n_days)) for t in time_steps) for n in nodes) / 1000
    dem_cool = sum(sum(sum(nodes[n]["cool"][d][t] * param["day_weights"][d] for d in range(n_days)) for t in time_steps) for n in nodes) / 1000
    
    T_cooling = sum(np.mean(nodes[n]["T_cooling_supply"]) for n in nodes)/len(nodes) + 273.15
    T_heating = sum(np.mean(nodes[n]["T_heating_supply"]) for n in nodes)/len(nodes) + 273.15
    T_ref = param["T_ref"] + 273.15
    
    for line in range(len(file)):
        if "gas_total" in file[line]:
            gas_total = float(str.split(file[line])[1])
    for line in range(len(file)):
        if "from_grid_total" in file[line]:
            from_grid_total = float(str.split(file[line])[1])
    for line in range(len(file)):
        if "to_grid_total" in file[line]:
            to_grid_total = float(str.split(file[line])[1])         
    # PV
    power_PV = read_energy_flow(file, "power_PV", "BU", param)
    pv_total = sum(sum(power_PV[d][t] for t in time_steps) * param["day_weights"][d] for d in range(n_days))

    # Air coolers
    if param["switch_bidirectional"]:
        cool_AIRC = {}
        for n in nodes:
            cool_AIRC[n] = read_energy_flow(file, "cool_AIRC", n, param)
        cool_AIRC["BU"] = read_energy_flow(file, "cool_AIRC", "BU", param)
        cool_AIRC["sum"] = np.sum(cool_AIRC[i] for i in cool_AIRC) / 1000
        airc_total = sum(sum(cool_AIRC["sum"][d][t]  for t in time_steps) * param["day_weights"][d] for d in range(n_days))    
    else:
        airc_total = 0
     
    
    # Calculate efficinecy
    # Note: exergy proportion of natural gas is set to 91,3% according to: "Efficiency analysis of a cogeneration and district energy system" by Rosen et al., DOI: 10.1016/j.applthermaleng.2004.05.008    
    dict_KPI["eta_ex"] = (dem_heat*(1-T_ref/T_heating) + dem_cool*(T_ref/T_cooling-1) + to_grid_total) / ( 0.913 * gas_total + from_grid_total + pv_total + airc_total*(T_ref/T_cooling-1))
    
    
    
    # Energetic System figure of merit (FOM)
    dict_KPI["FOM_system"] = (dem_heat + dem_cool + to_grid_total) / (gas_total + from_grid_total + pv_total)
   
    
    # CO2 - Emissions [kg/MWh_th]
    for line in range(len(file)):
        if "total_CO2" in file[line]:
            dict_KPI["co2_total"] = float(str.split(file[line])[1])
    dict_KPI["co2_spec"] = dict_KPI["co2_total"] * 1000 / dem_total      
    
    # Use of primary energy
    dict_KPI["PE_total"] = (from_grid_total - to_grid_total) * param["PEF_power"] + gas_total * param["PEF_gas"]
    dict_KPI["PE_spec"] = dict_KPI["PE_total"] / dem_total
    
    
    
    # Write txt
    with open(dir_results + "\KPIs.txt", "w") as outfile:
        outfile.write("KPIs of the bidirectional system \n")
        outfile.write("-------------------------------- \n")
        outfile.write(dir_results +  "\n\n\n")
        
        outfile.write("Demand Overlap Coefficients \n")
        outfile.write("--------------------------- \n")
        outfile.write("Demand DOC:    " + str(round(dict_KPI["DOC_dem"],3)) + "\n")
        outfile.write("Mean BES DOC:  " + str(round(dict_KPI["DOC_BES"]["sum"],3)) + "\n")
        outfile.write("Network DOC:   " + str(round(dict_KPI["DOC_N"],3)) + "\n\n\n")
        
        outfile.write("Economic KPIs \n")    
        outfile.write("------------- \n")   
        outfile.write("Total annualized costs [kEUR/a]:                    " + str(round(dict_KPI["tac_total"],2)) +"\n") 
        outfile.write("Specific thermal energy supply costs [EUR/MWh_th]:  " + str(round(dict_KPI["supply_costs"],3)) +"\n\n\n") 
        
        outfile.write("System efficiency \n")  
        outfile.write("----------------- \n")
        outfile.write("Exergetic efficiency:       " + str(round(dict_KPI["eta_ex"],3)) + "\n")
        outfile.write("Energetic figure of merit:  " + str(round(dict_KPI["FOM_system"],3)) + "\n\n\n")
    
        outfile.write("Ecological KPIs \n")  
        outfile.write("----------------- \n")
        outfile.write("Total CO2 emissions [t/a]:                 " + str(round(dict_KPI["co2_total"],2)) + "\n")
        outfile.write("Specific CO2 emissions [kg/MWh_th]:        " + str(round(dict_KPI["co2_spec"],2)) + "\n")
        outfile.write("Total use of primary energy [MWh_PE]:            " + str(round(dict_KPI["PE_total"],2)) + "\n") 
        outfile.write("Specific use of primary energy [MWh_PE/MWh_th]:  " + str(round(dict_KPI["PE_spec"],3)) + "\n")   
        
        
    with open(dir_results + "\System_KPIs.json", "w") as outfile:
        json.dump(dict_KPI, outfile, indent=4, sort_keys=True)   
        
        
    
    print("KPI calculation finished!")

#%%

def plot_costs(file, param, nodes, dir_costs):
    
    print("Creating cost structure plots...")
    
    
    all_devs = ["BOI", "CHP", "AC", "CC", "TES", "CTES", "AIRC", "HP", "EH", "BAT", "PV"] 
    
    if not param["switch_bidirectional"]:
        all_devs.append("SUB")
    
        
    all_devs_dom = ["HP", "CC", "EH", "FRC", "AIRC", "BOI", "TES"]  

    
    # Read costs 
    
    # BU devices
    c_devs = {}
    for dev in all_devs:
        string = "total_annual_costs_"+dev
        for line in range(len(file)):
            if string in file[line]:
                c_devs[dev] = float(str.split(file[line])[1]) 
                break
    c_devs_total = sum(c_devs[dev] for dev in all_devs)
    
    node_list = range(len(nodes))
    
    # Building devices
    c_devs_dom = {}
    for dev in all_devs_dom:
        c_devs_dom[dev] = 0
        string = "total_annual_costs_"+dev+"_n0"
        for line in range(len(file)):
            if string in file[line]:
                for n in node_list:
                    c_devs_dom[dev] += float(str.split(file[line+n])[1]) 
                break
    c_devs_dom_total = sum(c_devs_dom[dev] for dev in all_devs_dom)       
        
    # Gas costs
    gas = {}
    gas_vars = ["gas_total", "grid_limit_gas"]
    for item in gas_vars:
        for line in range(len(file)):
            if item in file[line]:
                gas[item] = float(str.split(file[line])[1]) 
                break
    c_gas = gas["gas_total"] * param["price_gas"]
    c_grid_gas = gas["grid_limit_gas"] * param["price_cap_gas"]
    c_gas_total = c_gas + c_grid_gas
    
    # Electricity costs
    elec = {}
    elec_vars = ["electricity_costs", "grid_limit_el"]
    for item in elec_vars:
        for line in range(len(file)):
            if item in file[line]:
                elec[item] = float(str.split(file[line])[1]) 
                break  
    c_el = elec["electricity_costs"]
    c_grid_el = elec["grid_limit_el"] * param["price_cap_el"]
    c_el_total = c_el + c_grid_el
    
    #feed-in revenue
    rev = {}
    for dev in["CHP", "PV"]:
        string = "revenue_feed_in_"+dev
        for line in range(len(file)):
            if string in file[line]:
                rev[dev] = float(str.split(file[line])[1]) 
                break
    rev_PV = rev["PV"]
    rev_CHP = rev["CHP"]
            
    # total tac
    tac_total = param["c_network"] + c_devs_total + c_devs_dom_total + c_gas_total + c_el_total - rev_PV - rev_CHP
            
    
    # Create plot            
    
    fig = plt.figure()  
    
    ax = fig.add_subplot(1,1,1, ylabel = "Annual cash-flow [kEUR]")
    width = 0.11
    # Plot costs
    ind = 0.33
    ax.bar(ind, param["c_network"], width)
    ax.bar(ind, c_devs_total,width, bottom = param["c_network"])
    ax.bar(ind, c_devs_dom_total,width, bottom=param["c_network"]+c_devs_total)
    ax.bar(ind, c_gas_total,width, bottom=param["c_network"]+c_devs_total+c_devs_dom_total)
    ax.bar(ind, c_el_total,width, bottom=param["c_network"]+c_devs_total+c_devs_dom_total+c_gas_total)
    # Plot revenue
    ind=0.66
    ax.bar(ind, rev_PV,width)
    ax.bar(ind, rev_CHP,width, bottom=rev_PV)
    
    ax.set_xlim(0,1)
    ax.set_xticks((0.33, 0.66))
    ax.set_xticklabels(("costs", "revenues"))
#    ax.legend((param["c_network"], c_devs_total,c_devs_dom_total,c_gas_total, c_el_total, rev_PV, rev_CHP), ("Network", "BU devices", "Building devices", "Gas", "Electricity", "PV feed-in", "CHP feed-in"))

    # Tag costs
    if param["c_network"] != 0:
        ax.text(0.15, param["c_network"]/2, "Network", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, param["c_network"]/2, str(round(param["c_network"],2)), horizontalalignment='center', verticalalignment='center')
    if c_devs_total != 0:
        ax.text(0.15, param["c_network"]+c_devs_total/2, "BU devices", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, param["c_network"]+c_devs_total/2, str(round(c_devs_total,2)), horizontalalignment='center', verticalalignment='center')
    if c_devs_dom_total != 0:
        ax.text(0.15, param["c_network"]+c_devs_total + c_devs_dom_total/2, "Building devices", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, param["c_network"]+c_devs_total + c_devs_dom_total/2, str(round(c_devs_dom_total,2)), horizontalalignment='center', verticalalignment='center')    
    if c_gas_total != 0:
        ax.text(0.15, param["c_network"]+c_devs_total + c_devs_dom_total + c_gas_total/2, "Gas", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, param["c_network"]+c_devs_total + c_devs_dom_total + c_gas_total/2, str(round(c_gas_total,2)), horizontalalignment='center', verticalalignment='center') 
    if c_el_total != 0:
        ax.text(0.15, param["c_network"]+c_devs_total + c_devs_dom_total + c_gas_total + c_el_total/2, "Electricity", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, param["c_network"]+c_devs_total + c_devs_dom_total + c_gas_total + c_el_total/2, str(round(c_el_total,2)), horizontalalignment='center', verticalalignment='center')
    
    # Tag revenues
    if rev_PV != 0:
        ax.text(0.82, rev_PV/2, "PV feed-in", horizontalalignment='center', verticalalignment='center')
        ax.text(0.66, rev_PV/2, str(round(rev["PV"],2)), horizontalalignment='center', verticalalignment='center')
    if rev_CHP != 0:
        ax.text(0.82, rev_PV + rev_CHP/2, "CHP feed-in", horizontalalignment='center', verticalalignment='center')
        ax.text(0.66, rev_PV + rev_CHP/2, str(round(rev["CHP"],2)), horizontalalignment='center', verticalalignment='center')
    
    # Tag total tac
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=y_max*1.1)
    ax.text(0.5,y_max*1.05,"TOTAL ANNUALIZED COSTS: " + str(round(tac_total,2)) + " kEUR/a", horizontalalignment='center', verticalalignment='center', fontweight = "bold")
    
    fig.savefig(fname = dir_costs + "\cost_structure.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)    
    plt.close(fig)

    print("All plots created!")

    
    
#%%    

def plot_sorted_curves(file, param, dir_grid):
    
    
    print("Creating sorted load curves...")

    if not os.path.exists(dir_grid):
        os.makedirs(dir_grid)   
          
    series = ["power_to_grid", "power_from_grid",
              "power_CHP", "power_PV",
              "feed_in_CHP", "feed_in_PV"]

    # Read grid feed-in    
    clustered = {}
    for item in series:
        clustered[item] = read_energy_flow(file, item, "BU", param)
    
    # Arrange full time series with 8760 steps
    full = {}
    for item in series:
        full[item] = np.zeros(8760)   
    # get List of days used as type-days
    z = param["day_matrix"]
    typedays = []
    for d in range(365):
        if any(z[d]):
            typedays.append(d)            
    # Arrange time series
    for d in range(365):
        match = np.where(z[:,d] == 1)[0][0]
        typeday = np.where(typedays == match)[0][0]        
        for item in series:
            full[item][24*d:24*(d+1)] = clustered[item][typeday,:]
        
    
    # Sort time series descending
    sort = {}
    for item in series:
        sort[item] = np.sort(full[item])[::-1]
    
    
    # Plot power from grid and power to grid   
    fig = plt.figure()        
    ax = fig.add_subplot(1,1,1, ylabel = "grid power [MW]", xlabel="Hours") 
    ax.plot(range(8760), sort["power_to_grid"], color="black", linewidth=2, label="power to grid")
    ax.plot(range(8760), sort["power_from_grid"], color="red", linewidth=2, label = "power from grid")
    ax.set_xlim(0,8760)
    ax.legend()
    ax.grid()   
    fig.savefig(fname = dir_grid + "\grid_sorted_curve.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)    
    plt.close()
 
    # Plot CHP load  
    fig = plt.figure()        
    ax = fig.add_subplot(1,1,1, ylabel = "CHP power [MW]", xlabel="Hours") 
    ax.plot(range(8760), sort["power_CHP"], color="green", linewidth=2, label="power CHP")
    ax.plot(range(8760), sort["power_PV"], color="yellow", linewidth=2, label="power PV")
    ax.set_xlim(0,8760)
    ax.legend()
    ax.grid()   
    fig.savefig(fname = dir_grid + "\CHP_sorted_curve.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)    
    plt.close()    
    

    print("All plots created!")    
    
    
#%%    


def plot_balancing_matrix(file, param, nodes, dir_balancing):
    
    
    print("Creating balancing matrices...")

    if not os.path.exists(dir_balancing):
        os.makedirs(dir_balancing)   
        
    time_steps = range(24)    
    n_days = param["n_clusters"]

    node_list = range(len(nodes))

    # Read node loads taken from grid
    res_nodes = {}
    for n in node_list:
        res_nodes[n] = post.read_energy_flow(file, "residual_thermal", n, param) / 1000
    # Read residual grid load in kW
    res_total = np.sum(res_nodes[n] for n in node_list) / 1000
    
    # Create dictionary for all balancing matrices
    matrix_dict = {}
    # Create array for balanced energy amounts
    balanced = np.zeros((n_days,len(time_steps)))

    # Day matrices
    for d in range(n_days):
        
        matrix_dict[d] = {}
       
        
        for t in time_steps:

            # Create balancing matrix for every time step
            matrix = np.zeros((len(nodes), len(nodes)))     

            # Collect heat consumers
            heat_nodes = []
            for n in node_list:
                if res_nodes[n][d][t] >= 0:
                    heat_nodes.append(n)
            # Collect cool consumers
            cool_nodes = []
            for n in node_list:
                if res_nodes[n][d][t] < 0:
                    cool_nodes.append(n)
                    
            # Check residual grid load
            if res_total[d][t] >= 0: # if residual heat demand               
                for c in cool_nodes:
                     # heat consumers (= cool producers) provide according to their heating demand
                    for h in heat_nodes:
                        matrix[c][h] = res_nodes[c][d][t] * res_nodes[h][d][t]/sum(res_nodes[h][d][t] for h in heat_nodes)
                        matrix[h][c] = - matrix[c][h]
            else: # if residual cooling demand
                for h in heat_nodes:
                    # cool consumers ( = heat producers) provide according to their cooling demand
                    for c in cool_nodes:
                        matrix[h][c] =  res_nodes[h][d][t] * res_nodes[c][d][t]/sum(res_nodes[c][d][t] for c in cool_nodes)
                        matrix[c][h] = - matrix[h][c]
            
            # Store matrix of current time step in dictionary
            matrix_dict[d][t] = matrix
            
            # Calculate balanced energy amount
            for n1 in node_list:
                for n2 in node_list:
                    if matrix[n1,n2] > 0:
                        balanced[d][t] += matrix[n1,n2]
            
#            
    # Sum up total heating and cooling balancing
    matrix_total = {}
    matrix_total["heat"] = np.zeros((len(nodes), len(nodes)))
    matrix_total["cool"] = np.zeros((len(nodes), len(nodes)))
    for d in range(n_days):
        for t in time_steps:
            matrix = matrix_dict[d][t] * param["day_weights"][d]
            for n1 in node_list:
                for n2 in node_list:
                    if matrix[n1][n2] >= 0:
                        matrix_total["heat"][n1][n2] += matrix[n1][n2]
                    else:
                        matrix_total["cool"][n1][n2] += matrix[n1][n2]
                        
    # Combined heating and cooling matrix
    matrix_total["total"] = np.zeros((len(nodes), len(nodes)))
    for n1 in node_list:
        for n2 in node_list:
            matrix_total["total"][n1][n2] = matrix_total["heat"][n1][n2] + matrix_total["cool"][n1][n2]

#    matrix_total["heat"] = 0.5*sum(sum((abs(matrix_dict[d][t]) + matrix_dict[d][t]) for t in time_steps) * param["day_weights"][d] for d in range(n_days))
    
#    # Calulate total balanced energy
    balanced_total = np.sum(matrix_total["heat"])
                                    

    # Create plots
   
    for dem in ["heat", "cool", "total"]:
       
        max_value = np.max( np.abs(matrix_total[dem]))        
    
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9)
        
        for row in range(len(nodes)):
            for col in range(len(nodes)):
                value = (np.round(matrix_total[dem][row,col], decimals=1))
                if value == 0:
                    ax.text(col, row, "x", va='center', ha='center', fontsize=9)
                else:
                    if abs(value) >= 0.7*max_value:
                        ax.text(col,row, str(value), va='center', ha='center', fontsize=9,color="white")
                    else:
                        ax.text(col,row, str(value), va='center', ha='center', fontsize=9,color="black")
        
    
        ax.matshow(matrix_total[dem], vmax=max_value, vmin=-max_value, cmap=plt.cm.seismic)
    
        
        ax.xaxis.tick_top()
        ax.set_ylabel("Verbraucher", fontsize=12, fontweight = "bold")     
        ax.set_xlabel("Erzeuger", fontsize=12, fontweight = "bold")  
        ax.xaxis.set_label_position('top')
        ticks = np.arange(len(nodes))
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        labels = ["{:2d}".format(x+1) for x in ticks]
        ax.set_xticklabels(labels, fontsize = 12)
        ax.set_yticklabels(labels, fontsize = 12)
    #    ax.set_title("Balancing between buildings [MWh]")
#        plt.show()
        
        
    #    fig.savefig(fname = dir_balancing + "\matrix_balancing.pdf", dpi = 600, format = "pdf", bbox_inches="tight", pad_inches=0.1)    
        fig.savefig(fname = dir_balancing + "\matrix_balancing_" + dem + ".pdf")
        plt.close()
    

            
        
    

    print("All plots created!")    

                    


#%%
    

def plot_flexibility(file, param, nodes, dir_sum):
    
    print("Creating building sum plots...")

    if not os.path.exists(dir_sum):
        os.makedirs(dir_sum)   
        
    time_steps = range(24)
    flow_colors =get_compColor()    
    n_days = param["n_clusters"]

    # List of all energy flows
    all_flows = ["heat_HP",
                 "heat_EH",
                 "heat_BOI",
                 "dch_TES", "ch_TES",
                 "soc_TES"
                 ]
    
    
    # list of all demands
    all_dem = ["heat",
               "cool"] 
    
    all_flows_BU =  ["heat_HP",
                     "heat_EH",
                     "heat_BOI",
                     "heat_CHP",
                     "dch_TES", "ch_TES", "soc_TES",
                     "cool_CC",
                     "cool_AC", "heat_AC",
                     "dch_CTES", "ch_CTES", "soc_CTES",
                     "cool_AIRC",
                     "power_PV"
                     ]
    
    # Building time series
    series = {} 
    for item in all_flows:
        if item == "soc_TES":
            series[item] = np.zeros((n_days, len(time_steps)+1))             
        else:                
            series[item] = np.zeros((n_days, len(time_steps))) 
    for item in all_dem:
        series[item+"_dem"] = np.zeros((n_days, len(time_steps)))
    for n in nodes:
        # Read all time series for node n
        for flow in all_flows:
            series[flow] += read_energy_flow(file, flow, n, param) / 1000
        for dem in all_dem:
            series[dem+"_dem"] += nodes[n][dem] / 1000
            
    # BU time series
    series_BU = {}
    for flow in all_flows_BU:
        series_BU[flow] = read_energy_flow(file, flow, "BU", param)

    # Read thermal residual loads and split them into cooling and heating load
    residual_thermal = read_energy_flow(file, "residual_thermal", "BU", param)
    series_BU["heat_dem"] = np.zeros((n_days, len(time_steps)))
    series_BU["cool_dem"] = np.zeros((n_days, len(time_steps)))
    for d in range(n_days):
        for t in time_steps:
            if residual_thermal[d][t] > 0:
                series_BU["heat_dem"][d][t] = residual_thermal[d][t]
            else:
                series_BU["cool_dem"][d][t] = - residual_thermal[d][t]
    series_BU["power_dem"] = read_energy_flow(file, "residual_power", "BU", param)            
    
    for d in range(n_days):
        
        fig = plt.figure()        
    
        # Plot sum of heating balances
        ax = fig.add_subplot(5,1,1, ylabel = "Heat [MW]")        
         # sources
        heat_sources = ["heat_HP", "heat_BOI", "heat_EH", "dch_TES"]
        plot_series = np.zeros((len(heat_sources), len(time_steps)))
        for k in range(len(heat_sources)):
            plot_series[k,:] = series[heat_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in heat_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = heat_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series["heat_dem"][d], label = "heat_dem", color = "k", linewidth = 2, zorder = -1)
        ax.step(time_steps, series["heat_dem"][d] + series["ch_TES"][d], label = "ch_TES", color = flow_colors["TES"], linewidth = 2, zorder = -10) 
        # Plot PV generation   
        ax.step(time_steps, series_BU["power_PV"][d], label = "power_PV", color = "orange")            
        ax.legend(loc='upper center', ncol=7, fontsize = 6) 
        y_max = ax.get_ylim()[1]
        ax.set_ylim(bottom=0, top=y_max*1.4)
        ax.set_yticks(np.arange(0,ax.get_ylim()[1],step=0.5))
        ax.set_xticklabels([])        
        
        # BU heating balance
        ax = fig.add_subplot(5,1,2, ylabel = "Heat [MW]")          
         # sources
        heat_sources = ["heat_CHP", "heat_HP", "heat_BOI", "dch_TES"]
        plot_series = np.zeros((len(heat_sources), len(time_steps)))
        for k in range(len(heat_sources)):
            plot_series[k,:] = series_BU[heat_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in heat_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = heat_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series_BU["heat_dem"][d], label = "heat_dem", color = "k", linewidth = 2, zorder = -1)
        ax.step(time_steps, series_BU["heat_dem"][d] + series_BU["heat_AC"][d], label = "heat_AC", color = flow_colors["AC"], linewidth = 2, zorder = -10)                
        ax.step(time_steps, series_BU["heat_dem"][d] + series_BU["heat_AC"][d] + series_BU["ch_TES"][d], label = "ch_TES", color = flow_colors["TES"], linewidth = 2, zorder = -20)
#        y_max = ax.get_ylim()
        ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.5)
        ax.legend(loc='upper center', ncol=7, fontsize = 6)     
        ax.set_xticklabels([])  
        
        # BU Cooling balance
        ax = fig.add_subplot(5,1,3, ylabel = "Cool [MW]")         
        # Sources
        cool_sources = ["cool_CC", "cool_AC", "cool_AIRC", "dch_CTES"]
        plot_series = np.zeros((len(cool_sources), len(time_steps)))
        for k in range(len(cool_sources)):
            plot_series[k,:] = series_BU[cool_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series_BU["cool_dem"][d], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
        ax.step(time_steps, series_BU["cool_dem"][d] + series_BU["ch_CTES"][d], label = "ch_CTES", color = flow_colors["CTES"], linewidth = 2, zorder = -10)
        ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.5)
        ax.legend(loc = "upper center", ncol= 6, fontsize = 6)         
        ax.set_xticklabels([])         
           
        # Sum of  TES and CTES soc's
        ax = fig.add_subplot(5,1,4, ylabel = "SOC [MWh]")  
        ax.step(range(25), series["soc_TES"][d], label = "soc_TES_buildings", color="r", linewidth=2)
        ax.step(range(25), series_BU["soc_TES"][d], label = "soc_TES_BU", color="orange", linewidth=2)
        ax.step(range(25), series_BU["soc_CTES"][d], label = "soc_CTES_BU", color="blue", linewidth=2)
        y_max = ax.get_ylim()[1]
        ax.set_ylim(bottom=0, top=y_max*1.45)
        ax.legend(loc='upper center', ncol=3, fontsize = 6) 
        ax.set_xticklabels([])  
        
        # Feed-in revenues
        ax = fig.add_subplot(5,1,5, ylabel = "EUR/MWh")  
        ax.step(time_steps, param["revenue_feed_in"]["PV"][d]*1000, color = "orange", label = "PV_revenue", linewidth=2)
        ax.step(time_steps, param["revenue_feed_in"]["CHP"][d]*1000, color = "green", label = "CHP_revenue", linewidth=2)
        ax.legend(loc='upper center', ncol=6, fontsize = 6) 
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=y_max*1.25)


        fig.subplots_adjust(hspace = 0.3)

        fig.savefig(fname = dir_sum + "\Day " +str(d)+".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
        plt.close(fig)     
        
        
    print("All plots created!")
        
        
        
        
#%%

def plot_storage_soc(file, param, dir_soc):
    
    print("Creating storage SOC plots...")

    if not os.path.exists(dir_soc):
        os.makedirs(dir_soc) 
    
    all_flows = ["soc_TES", "soc_CTES"]
    
    series = {}
    for flow in all_flows:
        series[flow] = read_soc(file, flow, "BU", param)

    time_steps = range(len(series["soc_TES"]))        
        
    fig = plt.figure()
       
    plt.step(time_steps, series["soc_TES"], color = "r", linewidth = 1)
    plt.step(time_steps, series["soc_CTES"], color = "b", linewidth = 0.2)
    plt.ylabel("Storage SOC [MWh]")
    plt.grid()

    fig.savefig(fname = dir_soc + "\soc_plot.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    
    print("All plots created!")
    
 
#%%

def plot_capacities(file, param, nodes, dir_caps):
    
    print("Creating capacity plots...")
    
    dev_colors = get_compColor()
    time_steps = range(24)
    n_days = param["n_clusters"]
    
    if not os.path.exists(dir_caps):
        os.makedirs(dir_caps)     
    
    if param["switch_bidirectional"]: 
        # Plot buildings devices
        all_devs_dom = ["HP", "EH","BOI", "CC", "FRC", "AIRC"]
        all_flows = ["heat_HP", "heat_EH", "heat_BOI", "cool_CC", "cool_FRC", "cool_AIRC"]
        
        ind = np.arange(len(all_devs_dom))
        plot_colors = tuple(dev_colors[dev] for dev in all_devs_dom)
        
        caps = {}
        for device in all_devs_dom:
            caps[device] = read_building_caps(file, device, nodes)
            
        node_list = range(len(nodes))
            
        for n in node_list:
            
            fig = plt.figure()
        
            # Plot device capacities
            ax = fig.add_subplot(2,1,1, ylabel = "Device cap [kW]")
            
            node_caps = tuple(caps[device][n] for device in all_devs_dom)
                    
            plot = plt.bar(ind, node_caps, color=plot_colors, edgecolor="k")
            plt.xticks(ind, all_devs_dom)
#            plt.legend(plot, all_devs_dom, loc="upper center",ncol=3, fontsize=7)
            # Tag bars with numeric values
            for bar in plot:
                height = bar.get_height()
                width = bar.get_width()
                if height > 0:
                    ax.text(bar.get_x() + width/2, 1.01*height, '{}'.format(np.round(height,2)), ha = "center", va='bottom')
            y_max = ax.get_ylim()[1]
            ax.set_ylim(top=1.2*y_max)
            
            
            # Plot device energy generation
            ax = fig.add_subplot(2,1,2, ylabel = "Device gen [MWh]")
            
            total = {}
            for flow in all_flows:
                series = read_energy_flow(file, flow, n, param)
                total[flow] = sum(sum(series[d][t] for t in time_steps) * param["day_weights"][d] for d in range(n_days)) / 1000
            
            gens = tuple(total[flow] for flow in all_flows)
            
            plot = plt.bar(ind, gens, color=plot_colors, edgecolor="k")
            plt.xticks(ind, all_devs_dom)   
            # Tag bars with numeric values
            for bar in plot:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + width/2, 1.01*height, '{}'.format(np.round(height,2)), ha = "center", va='bottom')
            y_max = ax.get_ylim()[1]
            ax.set_ylim(top=1.2*y_max)
            
            fig.savefig(fname = dir_caps + "\\" +str(n)+".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
        #            plt.show()
            plt.close(fig)
        
       
        
        
        # Plot sum of building devices    
        fig = plt.figure()
        
        # Plot device capacities
        ax = fig.add_subplot(2,1,1, ylabel = "Device cap sum [MW]")
        
        device_caps = tuple(sum(caps[device][n] for n in node_list)/1000 for device in all_devs_dom)
                    
        plot = plt.bar(ind, device_caps, color=plot_colors, edgecolor="k")
        plt.xticks(ind, all_devs_dom)
#        plt.legend(plot, all_devs_dom, loc="upper center",ncol=3, fontsize=7)
        # Tag bars with numeric values
        for bar in plot:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + width/2, 1.01*height, '{}'.format(np.round(height,2)), ha = "center", va='bottom')
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=1.2*y_max)
        
        
        # Plot device energy generation
        ax = fig.add_subplot(2,1,2, ylabel = "Device gen sum [MWh]")
        
        total = {}
        for flow in all_flows:
            total[flow] = 0
            for n in node_list:
                series = read_energy_flow(file, flow, n, param)
                total[flow] += sum(sum(series[d][t] for t in time_steps) * param["day_weights"][d] for d in range(n_days)) / 1000
            
        gens = tuple(total[flow] for flow in all_flows)
        
        plot = plt.bar(ind, gens, color=plot_colors, edgecolor="k")
        plt.xticks(ind, all_devs_dom)   
        # Tag bars with numeric values
        for bar in plot:
            height = bar.get_height()
            width = bar.get_width()
            if height > 0:
                ax.text(bar.get_x() + width/2, 1.01*height, '{}'.format(np.round(height,2)), ha = "center", va='bottom')
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=1.2*y_max)
        
        fig.savefig(fname = dir_caps + "\\Building_sum.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
        #            plt.show()
        plt.close(fig)    
        
        
    # plot BU capacities
    all_devs = ["BOI", "CHP", "HP", "EH", "CC", "AC", "AIRC", "PV"]
    all_flows = ["heat_BOI", "power_CHP", "heat_HP", "heat_EH", "cool_CC", "cool_AC", "cool_AIRC", "power_PV"]
    
    ind = np.arange(len(all_devs))
    plot_colors = tuple(dev_colors[dev] for dev in all_devs)
    
    
    fig = plt.figure()

    # Plot device capacities
    ax = fig.add_subplot(2,1,1, ylabel = "Device cap [MW]")  
    
    caps = {}
    for dev in all_devs:
        string = "nominal_capacity_"+dev
        for line in range(len(file)):
            if string in file[line]:
                caps[dev] = float(str.split(file[line])[1])
                break
            
    BU_caps = tuple(caps[dev] for dev in all_devs)
    
    plot = plt.bar(ind, BU_caps, color=plot_colors, edgecolor="k")
    plt.xticks(ind, ["BOI", "CHP", "HP", "EH", "CC", "AC", "AIRC", "PV"])
#    plt.legend(plot, ("BOI", "CHP", "HP", "EH", "CC", "AC", "AIRC"), loc="upper center",ncol=4, fontsize=7)
    # Tag bars with numeric values
    for bar in plot:
        height = bar.get_height()
        width = bar.get_width()
        if height > 0:
            ax.text(bar.get_x() + width/2, 1.01*height, '{}'.format(np.round(height,2)), ha = "center", va='bottom')
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=1.2*y_max)
    
    # Plot generations
    ax = fig.add_subplot(2,1,2, ylabel = "Device gen [MWh]")
    
    total = {}
    for flow in all_flows:
        series = read_energy_flow(file, flow, "BU", param)
        total[flow] = sum(sum(series[d][t] for t in time_steps) * param["day_weights"][d] for d in range(n_days))
    
    gens = tuple(total[flow] for flow in all_flows)
    
    plot = plt.bar(ind, gens, color=plot_colors, edgecolor="k")
    plt.xticks(ind, ["BOI", "CHP", "HP", "EH", "CC", "AC", "AIRC", "PV"])   
    # Tag bars with numeric values
    for bar in plot:
        height = bar.get_height()
        width = bar.get_width()
        if height > 0:
            ax.text(bar.get_x() + width/2, 1.01*height, '{}'.format(np.round(height,2)), ha = "center", va='bottom')
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=1.2*y_max)
    
    fig.savefig(fname = dir_caps + "\BU.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
    plt.close(fig)    
    
    
    print("All plots created!")
    
    
    
    
#%%    

def plot_load_charts(file, param, nodes, dir_plots):
    
    
    print("Creating load plots...")
    
    time_steps = range(24)
    n_days = param["n_clusters"]
    node_list = range(len(nodes))
    
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)
    
    # get loads
    load = {}
    
    # original demands
    load["dem"] = {}
    load["dem"]["heat"] = sum(nodes[n]["heat"] for n in nodes) / 1000
    load["dem"]["cool"] = sum(nodes[n]["cool"] for n in nodes) / 1000
    
   # BES sum demands
    load["BES_sum"] = {}  
    load["BES_sum"]["heat"] = np.zeros((n_days, len(time_steps)))
    load["BES_sum"]["cool"]  = np.zeros((n_days, len(time_steps)))
    # building residual loads
    res_nodes = {}
    for n in node_list:
        res_nodes[n] = post.read_energy_flow(file, "residual_thermal", n, param)
    # sum up buildings
    for d in range(n_days):       
        for t in time_steps:   
            # Collect heat consumers
            heat_nodes = []
            for n in node_list:
                if res_nodes[n][d][t] >= 0:
                    heat_nodes.append(n)
            # Collect cool consumers
            cool_nodes = []
            for n in node_list:
                if res_nodes[n][d][t] < 0:
                    cool_nodes.append(n)
            # sum up
            load["BES_sum"]["heat"][d][t] = np.sum(res_nodes[h][d][t] for h in heat_nodes)/1000
            load["BES_sum"]["cool"][d][t] = np.sum(-res_nodes[c][d][t] for c in cool_nodes)/1000
            
                
#    # Read residual grid load in kW
#    load["BES_sum"]["heat"] = np.zeros((n_days, len(time_steps)))
#    load["BES_sum"]["cool"]  = np.zeros((n_days, len(time_steps)))
#    for n in nodes:
#        res_thermal = post.read_energy_flow(file, "residual_thermal", n, param)
#        for dd in range(n_days):
#            for t in time_steps:
#                if res_thermal[dd][t] > 0:
#                    load["BES_sum"]["heat"][dd][t] += res_thermal[dd][t] / 1000
#                else:
#                    load["BES_sum"]["cool"][dd][t] += -res_thermal[dd][t] / 1000
    
    # BU loads
    load["BU"] = {}
    load["BU"]["heat"] = np.zeros((n_days, len(time_steps)))
    load["BU"]["cool"] = np.zeros((n_days, len(time_steps)))
    residual_thermal = np.sum(res_nodes[n] for n in node_list) / 1000
    for dd in range(n_days):
        for t in time_steps:
            if residual_thermal[dd][t] > 0:
                load["BU"]["heat"][dd][t] = residual_thermal[dd][t]
            else:
                load["BU"]["cool"][dd][t] = - residual_thermal[dd][t]    
    
    
#    # Day plots
#    for d in range(n_days):
#        
#        fig = plt.figure()
#    
#        # Plot total heating and cooling demands
#        ax = fig.add_subplot(3,1,1, ylabel = "Total dem [MW]")
#        ax.step(time_steps, load["dem"]["heat"][d], color = "r", label = "heating", linewidth = 2)
#        ax.step(time_steps, load["dem"]["cool"][d], color = "b", label = "cooling", linewidth = 2)
#        ax.grid()
#        ax.set_ylim(bottom=0)
#        ylim_total = ax.get_ylim()[1]
#                
#        # Plot heating and cooling demands to BES
#        ax = fig.add_subplot(3,1,2, ylabel = "BES dem [MW]")       
#        ax.step(time_steps, load["BES_sum"]["heat"][d], color = "r", label = "heating", linewidth = 2)
#        ax.step(time_steps, load["BES_sum"]["cool"][d], color = "b", label = "cooling", linewidth = 2)
#        ax.set_ylim(bottom=0, top = ylim_total)
#        ax.grid()     
#
#        # Plot residual BU loads                 
#        ax = fig.add_subplot(3,1,3, ylabel = "BU load [MW]")
#        ax.step(time_steps, load["BU"]["heat"][d], color = "r", label = "heating", linewidth = 2)
#        ax.step(time_steps, load["BU"]["cool"][d], color = "b", label = "cooling", linewidth = 2)
#        ax.set_ylim(bottom=0, top = ylim_total)
#        ax.grid()  
#        
#        fig.subplots_adjust(hspace = 0.3)
##        print(d)
#        fig.savefig(fname = dir_plots + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
##            plt.show()
#        plt.close(fig)
        
        
    
    # Year plot
    # arrange full time series with 8760 steps
    load_year = {}
    for item in load:
        load_year[item] = {}
        for dem in ["heat", "cool"]:
             load_year[item][dem] = np.zeros(8760)   
        # get List of days used as type-days
        z = param["day_matrix"]
        typedays = []
        for d in range(365):
            if any(z[d]):
                typedays.append(d)            
        # Arrange time series
        for d in range(365):
            match = np.where(z[:,d] == 1)[0][0]
            typeday = np.where(typedays == match)[0][0]        
            for dem in ["heat", "cool"]:
                load_year[item][dem][24*d:24*(d+1)] = load[item][dem][typeday,:]

    
    # Create plot
    fig = plt.figure()
    
    time_steps = range(8760)
    
    # Plot total heating and cooling demands
    ax = fig.add_subplot(3,1,1, ylabel = "Total dem [MW]")
    
    ax.step(time_steps, load_year["dem"]["heat"], color = "r", label = "heating", linewidth = 2)
    ax.step(time_steps, load_year["dem"]["cool"], color = "b", label = "cooling", linewidth = 2)
    ax.grid()
    ax.set_ylim(bottom=0)
    ylim_total = ax.get_ylim()[1]
            
    # Plot heating and cooling demands to BES
    ax = fig.add_subplot(3,1,2, ylabel = "BES dem [MW]")       
    ax.step(time_steps, load_year["BES_sum"]["heat"], color = "r", label = "heating", linewidth = 2)
    ax.step(time_steps, load_year["BES_sum"]["cool"], color = "b", label = "cooling", linewidth = 2)
    ax.set_ylim(bottom=0, top = ylim_total)
    ax.grid()     

    # Plot residual BU loads                 
    ax = fig.add_subplot(3,1,3, ylabel = "BU load [MW]")
    ax.step(time_steps, load_year["BU"]["heat"], color = "r", label = "heating", linewidth = 2)
    ax.step(time_steps, load_year["BU"]["cool"], color = "b", label = "cooling", linewidth = 2)
    ax.set_ylim(bottom=0, top = ylim_total)
    ax.grid()  
    
    fig.subplots_adjust(hspace = 0.3)
#        print(d)
    fig.savefig(fname = dir_plots + "\\1Year.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
    plt.close(fig)
        
        
        
    print("All Plots created!")


#%%

def plot_bldg_balancing(file, param, nodes, dir_plots):
    
    time_steps = range(24)
    flow_colors = get_compColor()    
    n_days = param["n_clusters"]
    
    # List of all energy flows
    all_flows = ["heat_HP", "power_HP",
                 "heat_EH", "power_EH",
                 "heat_BOI",
                 "dch_TES", "ch_TES",
                 "cool_CC", "power_CC",
                 "cool_FRC", 
                 "cool_AIRC",
                 ]
    
    # list of all demands
    all_dem = ["heat",
               "cool"]
                 
        
    
    # Create plots for every building and every day
    
    for n in nodes:
        
        dir_node = dir_plots + "\\Building " + str(n)
        if not os.path.exists(dir_node):
            os.makedirs(dir_node)  
            
        print("Creating day plots for building " + str(n) + "...")
        
        
        # Read all time series for node n
        series = {}
        for flow in all_flows:
            series[flow] = read_energy_flow(file, flow, n, param)
        for dem in all_dem:
            series[dem+"_dem"] = nodes[n][dem]
        
        
        # Create plots for every building
        for d in range(n_days):
            
            fig = plt.figure()
 #           plt.title("Building " + str(n) + " Day " + str(d))
        
            # Heating balance
            ax = fig.add_subplot(2,1,1, ylabel = "Heat [kW]")          
             # sources
            heat_sources = ["heat_HP", "heat_BOI", "heat_EH", "dch_TES"]
            plot_series = np.zeros((len(heat_sources), len(time_steps)))
            for k in range(len(heat_sources)):
                plot_series[k,:] = series[heat_sources[k]][d]
            plot_colors = tuple(flow_colors[flow] for flow in heat_sources)
            ax.stackplot(time_steps, plot_series, step="pre", labels = heat_sources, colors = plot_colors, zorder = -100)            
            # sinks
            ax.step(time_steps, series["heat_dem"][d], label = "heat_dem", color = "k", linewidth = 2, zorder = -1)
            ax.step(time_steps, series["heat_dem"][d] + series["ch_TES"][d], label = "ch_TES", color = flow_colors["TES"], linewidth = 2, zorder = -10)                
            ax.legend(loc='upper center', ncol=3, fontsize = 6)
            y_max = ax.get_ylim()[1]
            ax.set_ylim(top=y_max*1.5)
            
            # Cooling balance
            ax = fig.add_subplot(2,1,2, ylabel = "Cool [kW]")         
            # Sources
            cool_sources = ["cool_CC", "cool_FRC", "cool_AIRC"]
            plot_series = np.zeros((len(cool_sources), len(time_steps)))
            for k in range(len(cool_sources)):
                plot_series[k,:] = series[cool_sources[k]][d]
            plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
            ax.stackplot(time_steps, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
            # sinks
            ax.step(time_steps, series["cool_dem"][d], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
            ax.legend(loc = "upper center", ncol= 2, fontsize = 6) 
            y_max = ax.get_ylim()[1]
            ax.set_ylim(top=y_max*1.5)
                        
            fig.subplots_adjust(hspace = 0.2)
            fig.savefig(fname = dir_node + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
            plt.close(fig)
            
     
        
        
    # Building sum plots
    
    print("Creating plots for sum of buildings...")
    
    # sum up building time series
    series = {}
    for flow in all_flows:
        series[flow] = np.zeros((param["n_clusters"],24))
        for n in nodes:
            series[flow] += read_energy_flow(file, flow, n, param)
    for dem in all_dem:
        series[dem+"_dem"] = np.sum(nodes[n][dem] for n in nodes)
        
    dir_sum = dir_plots + "\\SUM"
    if not os.path.exists(dir_sum):
        os.makedirs(dir_sum)  
        
    
    # Create plot for every day
    for d in range(n_days):
        
        fig = plt.figure()
 #           plt.title("Building " + str(n) + " Day " + str(d))
    
        # Heating balance
        ax = fig.add_subplot(2,1,1, ylabel = "Heat [kW]")          
         # sources
        heat_sources = ["heat_HP", "heat_BOI", "heat_EH", "dch_TES"]
        plot_series = np.zeros((len(heat_sources), len(time_steps)))
        for k in range(len(heat_sources)):
            plot_series[k,:] = series[heat_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in heat_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = heat_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series["heat_dem"][d], label = "heat_dem", color = "k", linewidth = 2, zorder = -1)
        ax.step(time_steps, series["heat_dem"][d] + series["ch_TES"][d], label = "ch_TES", color = flow_colors["TES"], linewidth = 2, zorder = -10)                
        ax.legend(loc='upper center', ncol=3, fontsize = 6)
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=y_max*1.5)
        
        # Cooling balance
        ax = fig.add_subplot(2,1,2, ylabel = "Cool [kW]")         
        # Sources
        cool_sources = ["cool_CC", "cool_FRC", "cool_AIRC"]
        plot_series = np.zeros((len(cool_sources), len(time_steps)))
        for k in range(len(cool_sources)):
            plot_series[k,:] = series[cool_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series["cool_dem"][d], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
        ax.legend(loc = "upper center", ncol= 2, fontsize = 6) 
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=y_max*1.5)
                    
        fig.subplots_adjust(hspace = 0.2)
        fig.savefig(fname = dir_sum + "\Day" + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
        plt.close(fig)
     
        
        
    
    # Year plot of building sum
    
    # List containing cumulated number of days for every month
    months = range(12)
    days_sum = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    
     # Arrange clustered time series to full year with 8760 steps
    full_year = {}
    for flow in series:
        full_year[flow] = np.zeros(8760)   
    # get List of days used as type-days
    z = param["day_matrix"]
    typedays = []
    for d in range(365):
        if any(z[d]):
            typedays.append(d)            
    # Arrange time series
    for d in range(365):
        match = np.where(z[:,d] == 1)[0][0]
        typeday = np.where(typedays == match)[0][0]        
        for flow in series:
            full_year[flow][24*d:24*(d+1)] = series[flow][typeday,:]
            
    
    # Store monthly means of every time series in dictionary
    month_means = {}
    for flow in series:
        month_means[flow] = np.zeros(len(months))
        for m in months:
            if m == 0:
                month_means[flow][m] = np.mean(full_year[flow][0:24*days_sum[0]])
            else:
                month_means[flow][m] = np.mean(full_year[flow][24*days_sum[m-1]:24*days_sum[m]])
        
    
    # Create plot
    fig = plt.figure()

    # Heating balance
    ax = fig.add_subplot(2,1,1, ylabel = "Heat [kW]")          
     # sources
    heat_sources = ["heat_HP", "heat_BOI", "heat_EH", "dch_TES"]
    plot_series = np.zeros((len(heat_sources), len(months)))
    for k in range(len(heat_sources)):
        plot_series[k,:] = month_means[heat_sources[k]]
    plot_colors = tuple(flow_colors[flow] for flow in heat_sources)
    ax.stackplot(months, plot_series, step="pre", labels = heat_sources, colors = plot_colors, zorder = -100)            
    # sinks
    ax.step(months, month_means["heat_dem"], label = "heat_dem", color = "k", linewidth = 2, zorder = -1)
    ax.step(months, month_means["heat_dem"] + month_means["ch_TES"], label = "ch_TES", color = flow_colors["TES"], linewidth = 2, zorder = -10)                
    ax.legend(loc='upper center', ncol=3, fontsize = 6)
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=y_max*1.5)
    
    # Cooling balance
    ax = fig.add_subplot(2,1,2, ylabel = "Cool [kW]")         
    # Sources
    cool_sources = ["cool_CC", "cool_FRC", "cool_AIRC"]
    plot_series = np.zeros((len(cool_sources), len(months)))
    for k in range(len(cool_sources)):
        plot_series[k,:] = month_means[cool_sources[k]]
    plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
    ax.stackplot(months, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
    # sinks
    ax.step(months, month_means["cool_dem"], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
    ax.legend(loc = "upper center", ncol= 2, fontsize = 6) 
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=y_max*1.5)
                
    fig.subplots_adjust(hspace = 0.2)
    fig.savefig(fname = dir_sum + "\\1Months.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
    plt.close(fig)
    
    
            
    print("All plots created!")
#                
                
#%%
def plot_BU_balances(file, param, nodes, dir_plots):
    
    time_steps = range(24)
    flow_colors = get_compColor()    
    n_days = param["n_clusters"]
    
    # List of all energy flows
    all_flows = ["heat_HP", "power_HP",
                 "heat_EH", "power_EH",
                 "heat_BOI",
                 "heat_CHP", "power_CHP",
                 "dch_TES", "ch_TES",
                 "cool_CC", "power_CC",
                 "cool_AC", "heat_AC",
                 "dch_CTES", "ch_CTES",
                 "cool_AIRC",
                 "power_from_grid", "power_to_grid",
                 "power_PV",
                 "dch_BAT", "ch_BAT"
                 ]
        
 
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)  
        
    print("Creating day plots for balancing unit...")
    
    
    # Read all time series for node n
    series = {}
    for flow in all_flows:
        series[flow] = read_energy_flow(file, flow, "BU", param)
    
    # Read thermal residual loads and split them into cooling and heating load
    if param["switch_bidirectional"]:
        residual_thermal = read_energy_flow(file, "residual_thermal", "BU", param)
        series["heat_dem"] = np.zeros((n_days, len(time_steps)))
        series["cool_dem"] = np.zeros((n_days, len(time_steps)))
        for d in range(n_days):
            for t in time_steps:
                if residual_thermal[d][t] > 0:
                    series["heat_dem"][d][t] = residual_thermal[d][t]
                else:
                    series["cool_dem"][d][t] = - residual_thermal[d][t]
        series["power_dem"] = read_energy_flow(file, "residual_power", "BU", param)
    # in case of conventional DHC: residual thermal loads equal original demands
    else:
        series["heat_dem"] = np.array(param["dem_heat"])
        series["cool_dem"] = np.array(param["dem_cool"])
        series["power_dem"] = np.zeros((n_days, len(time_steps)))
        
    # Create hour plots for every type-day
    for d in range(n_days):
        
        fig = plt.figure()
 #           plt.title("Building " + str(n) + " Day " + str(d))
    
        # Heating balance
        ax = fig.add_subplot(3,1,1, ylabel = "Heat [MW]")          
         # sources
        heat_sources = ["heat_CHP", "heat_HP", "heat_BOI", "heat_EH", "dch_TES"]
        plot_series = np.zeros((len(heat_sources), len(time_steps)))
        for k in range(len(heat_sources)):
            plot_series[k,:] = series[heat_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in heat_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = heat_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series["heat_dem"][d], label = "heat_dem", color = "k", linewidth = 2, zorder = -1)
        ax.step(time_steps, series["heat_dem"][d] + series["heat_AC"][d], label = "heat_AC", color = flow_colors["AC"], linewidth = 2, zorder = -10)                
        ax.step(time_steps, series["heat_dem"][d] + series["heat_AC"][d] + series["ch_TES"][d], label = "ch_TES", color = flow_colors["TES"], linewidth = 2, zorder = -20)
#        y_max = ax.get_ylim()
        ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
        ax.legend(loc='upper center', ncol=4, fontsize = 5) 
        
        # Cooling balance
        ax = fig.add_subplot(3,1,2, ylabel = "Cool [MW]")         
        # Sources
        cool_sources = ["cool_CC", "cool_AC", "cool_AIRC", "dch_CTES"]
        plot_series = np.zeros((len(cool_sources), len(time_steps)))
        for k in range(len(cool_sources)):
            plot_series[k,:] = series[cool_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series["cool_dem"][d], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
        ax.step(time_steps, series["cool_dem"][d] + series["ch_CTES"][d], label = "ch_CTES", color = flow_colors["CTES"], linewidth = 2, zorder = -10)
        ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
        ax.legend(loc = "upper center", ncol= 3, fontsize = 5) 
                    
        # Power balance
        ax = fig.add_subplot(3,1,3, ylabel = "Power [MW]")         
        # Sources
        power_sources = ["power_CHP", "power_PV", "dch_BAT", "power_from_grid"]
        plot_series = np.zeros((len(power_sources), len(time_steps)))
        for k in range(len(power_sources)):
            plot_series[k,:] = series[power_sources[k]][d]
        plot_colors = tuple(flow_colors[flow] for flow in power_sources)
        ax.stackplot(time_steps, plot_series, step="pre", labels = power_sources, colors = plot_colors, zorder = -100)            
        # sinks
        ax.step(time_steps, series["power_dem"][d], label = "power_dem", color = "k", linewidth = 2, zorder = -1)
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d], label = "power_HP", color = flow_colors["HP"], linewidth = 2, zorder = -10)
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d] + series["power_CC"][d], label = "power_CC", color = flow_colors["CC"], linewidth = 2, zorder = -20)
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d] + series["power_CC"][d] + series["power_EH"][d], label = "power_EH", color = flow_colors["EH"], linewidth = 2, zorder = -30)
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d] + series["power_CC"][d] + series["power_EH"][d] + series["ch_BAT"][d], label = "ch_BAT", color = flow_colors["ch_BAT"], linewidth = 2, zorder = -40)
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d] + series["power_CC"][d] + series["power_EH"][d] + series["ch_BAT"][d] + series["power_to_grid"][d], label = "power_to_grid", color = flow_colors["power_to_grid"], linewidth = 2, zorder = -50)
        ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
        ax.legend(loc = "upper center", ncol= 5, fontsize = 5)         
        
        
        fig.subplots_adjust(hspace = 0.2)
        fig.savefig(fname = dir_plots + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
        plt.close(fig)
        
     
        
        
    # Create year plot using monthy mean values
    
    # List containing cumulated number of days for every month
    months = range(12)
    days_sum = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    
     # Arrange clustered time series to full year with 8760 steps
    full_year = {}
    for flow in series:
        full_year[flow] = np.zeros(8760)   
    # get List of days used as type-days
    z = param["day_matrix"]
    typedays = []
    for d in range(365):
        if any(z[d]):
            typedays.append(d)            
    # Arrange time series
    for d in range(365):
        match = np.where(z[:,d] == 1)[0][0]
        typeday = np.where(typedays == match)[0][0]        
        for flow in series:
            full_year[flow][24*d:24*(d+1)] = series[flow][typeday,:]
            
    
    # Store monthly means of every time series in dictionary
    month_means = {}
    for flow in series:
        month_means[flow] = np.zeros(len(months))
        for m in months:
            if m == 0:
                month_means[flow][m] = np.mean(full_year[flow][0:24*days_sum[0]])
            else:
                month_means[flow][m] = np.mean(full_year[flow][24*days_sum[m-1]:24*days_sum[m]])
    
            
    # plot yearly BU balance
    fig = plt.figure()

    # Heating balance
    ax = fig.add_subplot(3,1,1, ylabel = "Heat [MW]")          
     # sources
    heat_sources = ["heat_CHP", "heat_HP", "heat_BOI", "heat_EH", "dch_TES"]
    plot_series = np.zeros((len(heat_sources), len(months)))
    for k in range(len(heat_sources)):
        plot_series[k,:] = month_means[heat_sources[k]]
    plot_colors = tuple(flow_colors[flow] for flow in heat_sources)
    ax.stackplot(months, plot_series, step="pre", labels = heat_sources, colors = plot_colors, zorder = -100)            
    # sinks
    ax.step(months, month_means["heat_dem"], label = "heat_dem", color = "k", linewidth = 2, zorder = -1)
    ax.step(months, month_means["heat_dem"] + month_means["heat_AC"], label = "heat_AC", color = flow_colors["AC"], linewidth = 2, zorder = -10)                
    ax.step(months, month_means["heat_dem"] + month_means["heat_AC"] + month_means["ch_TES"], label = "ch_TES", color = flow_colors["TES"], linewidth = 2, zorder = -20)
#        y_max = ax.get_ylim()
    ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
    ax.legend(loc='upper center', ncol=4, fontsize = 5) 
    
    # Cooling balance
    ax = fig.add_subplot(3,1,2, ylabel = "Cool [MW]")         
    # Sources
    cool_sources = ["cool_CC", "cool_AC", "cool_AIRC", "dch_CTES"]
    plot_series = np.zeros((len(cool_sources), len(months)))
    for k in range(len(cool_sources)):
        plot_series[k,:] = month_means[cool_sources[k]]
    plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
    ax.stackplot(months, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
    # sinks
    ax.step(months, month_means["cool_dem"], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
    ax.step(months, month_means["cool_dem"] + month_means["ch_CTES"], label = "ch_CTES", color = flow_colors["CTES"], linewidth = 2, zorder = -10)
    ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
    ax.legend(loc = "upper center", ncol= 3, fontsize = 5) 
                
    # Power balance
    ax = fig.add_subplot(3,1,3, ylabel = "Power [MW]")         
    # Sources
    power_sources = ["power_CHP", "power_PV", "dch_BAT", "power_from_grid"]
    plot_series = np.zeros((len(power_sources), len(months)))
    for k in range(len(power_sources)):
        plot_series[k,:] = month_means[power_sources[k]]
    plot_colors = tuple(flow_colors[flow] for flow in power_sources)
    ax.stackplot(months, plot_series, step="pre", labels = power_sources, colors = plot_colors, zorder = -100)            
    # sinks
    ax.step(months, month_means["power_dem"], label = "power_dem", color = "k", linewidth = 2, zorder = -1)
    ax.step(months, month_means["power_dem"] + month_means["power_HP"], label = "power_HP", color = flow_colors["HP"], linewidth = 2, zorder = -10)
    ax.step(months, month_means["power_dem"] + month_means["power_HP"] + month_means["power_CC"], label = "power_CC", color = flow_colors["CC"], linewidth = 2, zorder = -20)
    ax.step(months, month_means["power_dem"] + month_means["power_HP"] + month_means["power_CC"] + month_means["power_EH"], label = "power_EH", color = flow_colors["EH"], linewidth = 2, zorder = -30)
    ax.step(months, month_means["power_dem"] + month_means["power_HP"] + month_means["power_CC"] + month_means["power_EH"] + month_means["ch_BAT"], label = "ch_BAT", color = flow_colors["ch_BAT"], linewidth = 2, zorder = -40)
    ax.step(months, month_means["power_dem"] + month_means["power_HP"] + month_means["power_CC"] + month_means["power_EH"] + month_means["ch_BAT"] + month_means["power_to_grid"], label = "power_to_grid", color = flow_colors["power_to_grid"], linewidth = 2, zorder = -50)
    ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
    ax.legend(loc = "upper center", ncol= 5, fontsize = 5)         
    
    
    fig.subplots_adjust(hspace = 0.2)
    fig.savefig(fname = dir_plots + "\\Months.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
    plt.close(fig)
            
    

    
    
    
        
#                
         
    print("All plots created!")
      

#%%          
                

def read_energy_flow(file, flow, node, param):
    
    
    if "soc" in flow:
        time_steps = range(25)
    else:
        time_steps = range(24)
        
    n_days = param["n_clusters"]
    flows = np.zeros((n_days, len(time_steps)))  

    
    if node == "BU": # Read BU energy flows        
        string = flow + "_d0"
    else: # Read building energy flows    
        string = flow + "_n" + str(node)
        
    # Find first flow entry    
    for line in range(len(file)):
        if string in file[line]:
            line_0 = line
            break
            
    # Read flows
    for d in range(n_days):
        for t in time_steps:
            value = float(str.split(file[line_0 + len(time_steps)*d + t])[1])
            flows[d][t] = value
    
    return flows

#%%
def read_soc(file, flow, node, param):
    
    
    if node == "BU":
        time_steps = range(8761)
    else:
        time_steps = range(25)
        
    flows = np.zeros(len(time_steps)) 

    
    if node == "BU": # Read BU energy flows        
        string = flow + "_d0"
    else: # Read building energy flows    
        string = flow + "_n" + str(node)
        
    # Find first flow entry    
    for line in range(len(file)):
        if string in file[line]:
            line_0 = line
            break
            
    # Read flows
    for t in time_steps:
        value = float(str.split(file[line_0 + t])[1])
        flows[t] = value
    
    return flows


#%%

def read_building_caps(file, device, nodes):
    
    caps = np.zeros(len(nodes))
    
    string = "nominal_capacity_"+device+"_n"
    
    # Find first cap entry    
    for line in range(len(file)):
        if string in file[line]:
            line_0 = line
            break
        
    # Read caps
    for n in range(len(nodes)):
        caps[n] = float(str.split(file[line_0 + n])[1])

    return caps


#%%
    
    
def get_compColor():
    """
    This function defines a color for each device that is used for plots.
    
    """
    
    compColor = {}
       
    compColor["BOI"] = (0.843, 0.059, 0.059, 0.8)
    compColor["heat_BOI"] = compColor["BOI"]
    
    compColor["CHP"] = (0.137, 0.706, 0.196, 0.8)
    compColor["heat_CHP"] = compColor["CHP"]
    compColor["power_CHP"] = compColor["CHP"]
    
    compColor["EH"] = (0.961, 0.412, 0.412, 0.8)
    compColor["heat_EH"] = compColor["EH"]
    compColor["power_EH"] = compColor["EH"]
    
    compColor["HP"] = (0.471, 0.843, 1.0, 0.8)
    compColor["heat_HP"] = compColor["HP"]
    compColor["power_HP"] = compColor["HP"]

    compColor["PV"] = (1.000, 0.725, 0.000, 0.8)
    compColor["power_PV"] = compColor["PV"]

    compColor["AC"] = (0.529, 0.706, 0.882, 0.8)
    compColor["cool_AC"] = compColor["AC"]
    compColor["heat_AC"] = compColor["AC"]

    compColor["CC"] = (0.184, 0.459, 0.710, 0.8)
    compColor["cool_CC"] = compColor["CC"]
    compColor["power_CC"] = compColor["CC"]

    compColor["TES"] = (0.482, 0.482, 0.482, 0.8)
    compColor["ch_TES"] = compColor["TES"]
    compColor["dch_TES"] = compColor["TES"]

    compColor["CTES"] = (0.482, 0.482, 0.482, 0.8)
    compColor["ch_CTES"] = compColor["CTES"]
    compColor["dch_CTES"] = compColor["CTES"]
    
    compColor["BAT"] = (0.382, 0.382, 0.382, 0.8)   
    compColor["ch_BAT"] = compColor["BAT"]
    compColor["dch_BAT"] = compColor["BAT"]
    
    compColor["FRC"] = (0.529, 0.706, 0.882, 0.8)
    compColor["cool_FRC"] = compColor["FRC"]
    
    compColor["AIRC"] = (0.749, 0.749, 0.749, 1)
    compColor["cool_AIRC"] = compColor["AIRC"]
    
    compColor["HYB"] = compColor["HP"]
    compColor["cool_HYB"] = compColor["HYB"]    
    
    compColor["power_from_grid"] = (0.749, 0.749, 0.749, 1)
    compColor["power_to_grid"] = (0.749, 0.749, 0.749, 1)

    
    return compColor
