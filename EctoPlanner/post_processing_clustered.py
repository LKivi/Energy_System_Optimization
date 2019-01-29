# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19

@author: lki
"""


import matplotlib.pyplot as plt
import os
import numpy as np
#import time


#%%

# param und nodes als file!!!!!!
def run(dir_results, param, nodes):
    
    
     
    # Read solution file
    file_name = dir_results + "\\model.sol"
    with open(file_name, "r") as solution_file:
        file = solution_file.readlines()

    # Create folder for plots
    dir_plots = dir_results + "\\Plots"
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)
    
#    # Building balance plots
#    plot_bldg_balancing(file, param, nodes, dir_plots + "\\building_balances")
    
#    # BU balance plots
#    plot_BU_balances(file, param, dir_plots + "\\BU_balances")
    
    
#    # load plots
#    plot_load_charts(file, param, nodes, dir_plots + "\\load_charts")
    
    
#    # Plot capacities and generation
#    plot_capacities(file, param, nodes, dir_plots + "\\capacity_plots")
    
#    # Plot BU storage SOCs
#    plot_storage_soc(file, param, dir_plots + "\\soc_storages")
    
#    # Plot building sum
#    plot_flexibility(file, param, nodes, dir_plots + "\\flexibility")
    
#    # Plot sorted annual load curve of grid interaction
#    plot_grid_interaction(file, param, dir_plots + "\\sorted_load_curves")

#    # Plot cost structure
    plot_costs(file, param, nodes, dir_plots)



def calc_KPIs(file, nodes, param):
    
    time_steps = range(24)
    n_days = param["n_clusters"]
    
    
    # Cost KPIs
    # Costs for thermal energy supply (EUR/MWh)
    # Read total annualized costs (EUR)
    for line in range(len(file)):
        if "total_annualized_costs" in file[line]:
            tac_total = float(str.split(file[line])[1]) * 1000
            break            
    # total thermal energy demand (heating and cooling) (MWh)
    dem_total = sum(sum(sum(sum((nodes[n][dem][d][t] * param["day_weights"][d]) for dem in ["heat", "cool"]) for t in time_steps) for d in range(n_days)) for n in nodes) / 1000    
    # Calculate supply costs
    supply_costs = tac_total / dem_total
    
    # Costs for electricity generation
    
    
    
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
    DOC_dem = ( 2 * sum(sum( min( dem["heat"][d][t], dem["cool"][d][t] ) * param["day_weights"][d] for d in range(n_days)) for t in time_steps)) / (dem_total*1000)
    
    
    # Internal
    
    
    # Network
    # Sum up building residual loads
    subs_heating_demand = np.zeros((n_days, len(time_steps)))
    subs_cooling_demand = np.zeros((n_days, len(time_steps)))
    for n in nodes:
        res_thermal = post.read_energy_flow(file, "residual_thermal", n, param)
        for d in range(n_days):
            for t in time_steps:
                if res_thermal[d][t] > 0:
                    subs_heating_demand[d][t] += res_thermal[d][t]
                else:
                    subs_cooling_demand[d][t] += -res_thermal[d][t] 
    DOC_N = ( 2 * sum(sum( min(subs_heating_demand[d][t], subs_cooling_demand[d][t]) * param["day_weights"][d] for d in range(n_days)) for t in time_steps)) / (sum(sum((subs_heating_demand[d][t] + subs_cooling_demand[d][t]) * param["day_weights"][d] for d in range(n_days)) for t in time_steps))                      
    

#%%

def plot_costs(file, param, nodes, dir_costs):
    
    
    all_devs = ["BOI", "CHP", "AC", "CC", "TES", "CTES", "HYB", "HP", "EH", "BAT"] 
        
    all_devs_dom = ["HP", "CC", "EH", "FRC", "AIR", "BOI", "PV", "TES"]  

    
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
    
    # Building devices
    c_devs_dom = {}
    for dev in all_devs_dom:
        c_devs_dom[dev] = 0
        string = "total_annual_costs_"+dev+"_n0"
        for line in range(len(file)):
            if string in file[line]:
                for n in nodes:
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
    tac_total = c_devs_total + c_devs_dom_total + c_gas_total + c_el_total - rev_PV - rev_CHP
            
    
    # Create plot            
    
    fig = plt.figure()  
    
    ax = fig.add_subplot(1,1,1, ylabel = "Annual cash-flow [kEUR]")
    width = 0.11
    # Plot costs
    ind = 0.33
    ax.bar(ind, c_devs_total,width)
    ax.bar(ind, c_devs_dom_total,width, bottom=c_devs_total)
    ax.bar(ind, c_gas_total,width, bottom=c_devs_total+c_devs_dom_total)
    ax.bar(ind, c_el_total,width, bottom=c_devs_total+c_devs_dom_total+c_gas_total)
    # Plot revenue
    ind=0.66
    ax.bar(ind, rev_PV,width)
    ax.bar(ind, rev_CHP,width, bottom=rev_PV)
    
    ax.set_xlim(0,1)
    ax.set_xticks((0.33, 0.66))
    ax.set_xticklabels(("costs", "revenues"))
    ax.legend((c_devs_total,c_devs_dom_total,c_gas_total, c_el_total, rev_PV, rev_CHP), ("BU devices", "Building devices", "Gas", "Electricity", "PV feed-in", "CHP feed-in"))

    # Tag costs
    if c_devs_total != 0:
        ax.text(0.15, c_devs_total/2, "BU devices", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, c_devs_total/2, str(round(c_devs_total,2)), horizontalalignment='center', verticalalignment='center')
    if c_devs_dom_total != 0:
        ax.text(0.15, c_devs_total + c_devs_dom_total/2, "Building devices", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, c_devs_total + c_devs_dom_total/2, str(round(c_devs_dom_total,2)), horizontalalignment='center', verticalalignment='center')    
    if c_gas_total != 0:
        ax.text(0.15, c_devs_total + c_devs_dom_total + c_gas_total/2, "Gas", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, c_devs_total + c_devs_dom_total + c_gas_total/2, str(round(c_gas_total,2)), horizontalalignment='center', verticalalignment='center') 
    if c_el_total != 0:
        ax.text(0.15, c_devs_total + c_devs_dom_total + c_gas_total + c_el_total/2, "Electricity", horizontalalignment='center', verticalalignment='center')
        ax.text(0.33, c_devs_total + c_devs_dom_total + c_gas_total + c_el_total/2, str(round(c_el_total,2)), horizontalalignment='center', verticalalignment='center')
    
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
    
    
    
#%%    

def plot_grid_interaction(file, param, dir_grid):
    
    
    print("Creating grid interaction plots...")

    if not os.path.exists(dir_grid):
        os.makedirs(dir_grid)   
          
    series = ["power_to_grid", "power_from_grid"]

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
        
    
    # Sort grid feed-in descending
    to_grid_sorted = np.sort(full["power_to_grid"])[::-1]
    from_grid_sorted = np.sort(full["power_from_grid"])[::-1]
    
   
    fig = plt.figure()        

    # Plot sum of heating balances
    ax = fig.add_subplot(1,1,1, ylabel = "Feed-in [MW]", xlabel="Hours") 
    ax.plot(range(8760), to_grid_sorted, color="black", linewidth=2, label="power to grid")
    ax.plot(range(8760), from_grid_sorted, color="red", linewidth=2, label = "power from grid")
    ax.set_xlim(0,8760)
    ax.legend()
    ax.grid()
#    plt.show()
    
    fig.savefig(fname = dir_grid + "\grid_sorted_curve.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    
    plt.close()
    
    
    
    
#%%    


def plot_balancing_matrix(file, param, nodes, dir_balancing):
    
    
    print("Creating balancing matrices...")

    if not os.path.exists(dir_balancing):
        os.makedirs(dir_balancing)   
        
    time_steps = range(24)    
    n_days = param["n_clusters"]

    # Read node loads taken from grid
    res_nodes = {}
    for n in nodes:
        res_nodes[n] = read_energy_flow(file, "residual_thermal", n, param)
    # Read residual grid load in kW
    res_total = read_energy_flow(file, "residual_thermal", "BU", param) * 1000

    # Day matrices
    for d in range(n_days):
                
        # Create balancing matrix
        matrix = np.zeros((len(nodes), len(nodes)))
        
        for t in time_steps:
            
            # Collect heat consumers
            heat_nodes = []
            for n in nodes:
                if res_nodes[n][d][t] >= 0:
                    heat_nodes.append(n)
            # Collect cool consumers
            cool_nodes = []
            for n in nodes:
                if not n in heat_nodes:
                    cool_nodes.append(n)
                    
            # Check residual grid load
            if res_total[d][t] >= 0: # if residual heat demand               
                for h in heat_nodes:
                    # heat from BU to node h
                    heat_from_BU = res_total[d][t] * res_nodes[h][d][t]/(sum(res_nodes[h][d][t] for h in heat_nodes))
#                    matrix[h][len(nodes)] = heat_from_BU
                     # cool consumers (= heat producers) provide according to their cooling demand
                    for c in cool_nodes:
                        matrix[h][c] = (res_nodes[h][d][t] - heat_from_BU) * res_nodes[c][d][t]/(sum(res_nodes[c][d][t] for c in cool_nodes))
                        matrix[c][h] = -matrix[h][c]
            else: # if residual cooling demand
                for c in cool_nodes:
                    # Cooling from BU to node c
                    cool_from_BU = res_total[d][t] * res_nodes[c][d][t]/(sum(res_nodes[c][d][t] for c in cool_nodes))
#                    matrix[c][len(nodes)] = cool_from_BU
                    # heat consumers ( = cool producers) provide according to their heating demand
                    for h in heat_nodes:
                        matrix[c][h] = (res_nodes[c][d][t] - cool_from_BU) * res_nodes[h][d][t]/(sum(res_nodes[h][d][t] for h in heat_nodes))
                        matrix[h][c] = - matrix[c][h] 
        
#            for n in nodes:
#                matrix[n][len(nodes)+1] = sum(matrix[n][col] for col in range(len(nodes)+1))
                
            max_value = np.max( np.abs(matrix))        
        
            fig, ax = plt.subplots()
            
            for row in range(len(nodes)):
                for col in range(len(nodes)):
                    value = (np.round(matrix[row,col], decimals=2))
                    if value == 0:
                        ax.text(col, row, "--", va='center', ha='center', fontsize=7)
                    else:
                        if abs(value) >= 0.7*max_value:
                            ax.text(col, row, str(value), va='center', ha='center', fontsize=7,color="white")
                        else:
                            ax.text(col, row, str(value), va='center', ha='center', fontsize=7,color="black")
            

            ax.matshow(matrix, vmax=max_value, vmin=-max_value, cmap=plt.cm.seismic)

            
            ax.xaxis.tick_top()
            ax.set_ylabel("TO")     
            ax.set_xlabel("FROM")  
            ax.xaxis.set_label_position('top')
            ax.set_yticks(range(len(nodes)))
            ax.set_xticks(range(len(nodes)))
#            labels = [None]*(len(nodes))
#            labels[0:len(nodes)] = range(len(nodes))
#            labels[len(nodes)] = "BU"
#            labels[len(nodes)+1] = "SUM"
#            ax.set_xticklabels(labels)
#            ax.grid()

                    


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
                     "cool_HYB",
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
        cool_sources = ["cool_CC", "cool_AC", "cool_HYB", "dch_CTES"]
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
    
    time_steps = range(25)
    n_days = param["n_clusters"]

    if not os.path.exists(dir_soc):
        os.makedirs(dir_soc) 
    
    all_flows = ["soc_TES", "soc_CTES"]
    
    series = {}
    for flow in all_flows:
        series[flow] = read_energy_flow(file, flow, "BU", param)
        
    for d in range(n_days):
        
        fig = plt.figure()
        
        plt.step(time_steps, series["soc_TES"][d], color = "r", linewidth = 3)
        plt.step(time_steps, series["soc_CTES"][d], color = "b", linewidth = 3)
        plt.ylabel("Storage SOC [MWh]")
        plt.grid()
    
        fig.savefig(fname = dir_soc + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
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
    
    # Plot buildings devices
    all_devs_dom = ["HP", "EH","BOI", "CC", "FRC", "AIR"]
    all_flows = ["heat_HP", "heat_EH", "heat_BOI", "cool_CC", "cool_FRC", "cool_AIR"]
    
    ind = np.arange(len(all_devs_dom))
    plot_colors = tuple(dev_colors[dev] for dev in all_devs_dom)

    caps = {}
    for device in all_devs_dom:
        caps[device] = read_building_caps(file, device, nodes)
    
    
    
    for n in nodes:
        
        fig = plt.figure()
    
        # Plot device capacities
        ax = fig.add_subplot(2,1,1, ylabel = "Device cap [kW]")
        
        node_caps = tuple(caps[device][n] for device in all_devs_dom)
                
        plot = plt.bar(ind, node_caps, color=plot_colors, edgecolor="k")
        plt.xticks(ind, all_devs_dom)
        plt.legend(plot, all_devs_dom, loc="upper center",ncol=3, fontsize=7)
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=1.25*y_max)
        
        
        # Plot device energy generation
        fig.add_subplot(2,1,2, ylabel = "Device gen [MWh]")
        
        total = {}
        for flow in all_flows:
            series = read_energy_flow(file, flow, n, param)
            total[flow] = sum(sum(series[d][t] for t in time_steps) * param["day_weights"][d] for d in range(n_days)) / 1000
        
        gens = tuple(total[flow] for flow in all_flows)
        
        plot = plt.bar(ind, gens, color=plot_colors, edgecolor="k")
        plt.xticks(ind, all_devs_dom)   
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=1.25*y_max)
        
        fig.savefig(fname = dir_caps + "\\" +str(n)+".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
        plt.close(fig)
        
        
    # plot BU capacities
    all_devs = ["BOI", "CHP", "HP", "EH", "CC", "AC", "HYB"]
    all_flows = ["heat_BOI", "power_CHP", "heat_HP", "heat_EH", "cool_CC", "cool_AC", "cool_HYB"]
    
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
    plt.xticks(ind, ["BOI", "CHP", "HP", "EH", "CC", "AC", "AIR"])
    plt.legend(plot, ("BOI", "CHP", "HP", "EH", "CC", "AC", "AIR"), loc="upper center",ncol=4, fontsize=7)
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=1.25*y_max)
    
    # Plot generations
    ax = fig.add_subplot(2,1,2, ylabel = "Device gen [MWh]")
    
    total = {}
    for flow in all_flows:
        series = read_energy_flow(file, flow, "BU", param)
        total[flow] = sum(sum(series[d][t] for t in time_steps) * param["day_weights"][d] for d in range(n_days))
    
    gens = tuple(total[flow] for flow in all_flows)
    
    plot = plt.bar(ind, gens, color=plot_colors, edgecolor="k")
    plt.xticks(ind, ["BOI", "CHP", "HP", "EH", "CC", "AC", "AIR"])   
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=1.25*y_max)
    
    fig.savefig(fname = dir_caps + "\BU.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
    plt.close(fig)    
    
    
    print("All plots created!")
    
    
    
    
#%%    

def plot_load_charts(file, param, nodes, dir_plots):
    
    
    print("Creating load plots...")
    
    time_steps = range(24)
    n_days = param["n_clusters"]
    
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots) 
    
    
    for d in range(n_days):
        
        fig = plt.figure()
    
        # Plot total heating and cooling demands
        ax = fig.add_subplot(3,1,1, ylabel = "Total dem [MW]")
        
        total_heating_demand = sum(nodes[n]["heat"][d] for n in nodes) / 1000
        total_cooling_demand = sum(nodes[n]["cool"][d] for n in nodes) / 1000
        ax.step(time_steps, total_heating_demand, color = "r", label = "heating", linewidth = 2)
        ax.step(time_steps, total_cooling_demand, color = "b", label = "cooling", linewidth = 2)
        ax.grid()
        ax.set_ylim(bottom=0)
        ylim_total = ax.get_ylim()[1]
        
        
        # Plot heating and cooling demands at building substations
        ax = fig.add_subplot(3,1,2, ylabel = "Sub dem [MW]")
        # Sum up building loads
        subs_heating_demand = np.zeros((n_days, len(time_steps)))
        subs_cooling_demand = np.zeros((n_days, len(time_steps)))
        for n in nodes:
            res_thermal = read_energy_flow(file, "residual_thermal", n, param)
            for dd in range(n_days):
                for t in time_steps:
                    if res_thermal[dd][t] > 0:
                        subs_heating_demand[dd][t] += res_thermal[dd][t]
                    else:
                        subs_cooling_demand[dd][t] += -res_thermal[dd][t]
        ax.step(time_steps, subs_heating_demand[d] / 1000, color = "r", label = "heating", linewidth = 2)
        ax.step(time_steps, subs_cooling_demand[d] / 1000, color = "b", label = "cooling", linewidth = 2)
        ax.set_ylim(bottom=0, top = ylim_total)
        ax.grid()     

        # Plot residual BU loads                 
        ax = fig.add_subplot(3,1,3, ylabel = "BU load [MW]")
        residual_heating = np.zeros((n_days, len(time_steps)))
        residual_cooling = np.zeros((n_days, len(time_steps)))
        residual_thermal = read_energy_flow(file, "residual_thermal", "BU", param)
        for dd in range(n_days):
            for t in time_steps:
                if residual_thermal[dd][t] > 0:
                    residual_heating[dd][t] = residual_thermal[dd][t]
                else:
                    residual_cooling[dd][t] = - residual_thermal[dd][t]
        ax.step(time_steps, residual_heating[d], color = "r", label = "heating", linewidth = 2)
        ax.step(time_steps, residual_cooling[d], color = "b", label = "cooling", linewidth = 2)
        ax.set_ylim(bottom=0, top = ylim_total)
        ax.grid()  
        
        fig.subplots_adjust(hspace = 0.3)
#        print(d)
        fig.savefig(fname = dir_plots + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
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
                 "cool_AIR",
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
            cool_sources = ["cool_CC", "cool_FRC", "cool_AIR"]
            plot_series = np.zeros((len(cool_sources), len(time_steps)))
            for k in range(len(cool_sources)):
                plot_series[k,:] = series[cool_sources[k]][d]
            plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
            ax.stackplot(time_steps, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
            # sinks
            ax.step(time_steps, series["cool_dem"][d], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
            ax.legend(loc = "lower center", ncol= 2) 
            y_max = ax.get_ylim()[1]
            ax.set_ylim(top=y_max*1.5)
                        
            fig.subplots_adjust(hspace = 0.2)
            fig.savefig(fname = dir_node + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
            plt.close(fig)
            
    print("All plots created!")
#                
                
#%%
def plot_BU_balances(file, param, dir_plots):
    
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
                 "cool_HYB",
                 "power_from_grid", "power_to_grid",
                 "power_PV",
                 "dch_BAT", "ch_BAT"
                 ]
        
    
    # Create plots for every day    
    if not os.path.exists(dir_plots):
        os.makedirs(dir_plots)  
        
    print("Creating day plots for balancing unit...")
    
    
    # Read all time series for node n
    series = {}
    for flow in all_flows:
        series[flow] = read_energy_flow(file, flow, "BU", param)
    
    # Read thermal residual loads and split them into cooling and heating load
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
    
    
    # Create plots for BU
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
        cool_sources = ["cool_CC", "cool_AC", "cool_HYB", "dch_CTES"]
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
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d] + series["power_CC"][d] + series["power_EH"][d] + series["ch_BAT"][d][t], label = "ch_BAT", color = flow_colors["ch_BAT"], linewidth = 2, zorder = -40)
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d] + series["power_CC"][d] + series["power_EH"][d] + series["ch_BAT"][d][t] + series["power_to_grid"][d], label = "power_to_grid", color = flow_colors["power_to_grid"], linewidth = 2, zorder = -50)
        ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
        ax.legend(loc = "upper center", ncol= 5, fontsize = 5)         
        
        
        fig.subplots_adjust(hspace = 0.2)
        fig.savefig(fname = dir_plots + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
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

    compColor["CTES"] = (0.582, 0.582, 0.582, 0.8)
    compColor["ch_CTES"] = compColor["CTES"]
    compColor["dch_CTES"] = compColor["CTES"]
    
    compColor["BAT"] = (0.382, 0.382, 0.382, 0.8)   
    compColor["ch_BAT"] = compColor["BAT"]
    compColor["dch_BAT"] = compColor["BAT"]
    
    compColor["FRC"] = (0.529, 0.706, 0.882, 0.8)
    compColor["cool_FRC"] = compColor["FRC"]
    
    compColor["AIR"] = compColor["CTES"]
    compColor["cool_AIR"] = compColor["AIR"]
    
    compColor["HYB"] = compColor["HP"]
    compColor["cool_HYB"] = compColor["HYB"]    
    
    compColor["power_from_grid"] = (0.749, 0.749, 0.749, 1)
    compColor["power_to_grid"] = (0.749, 0.749, 0.749, 1)

    
    return compColor
