# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19

@author: lki
"""


import matplotlib.pyplot as plt
import os
import numpy as np
#import time




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
    
    # Building balance plots
    plot_bldg_balancing(file, param, nodes, dir_plots + "\\building_balances")
    
    # BU balance plots
    plot_BU_balances(file, param, dir_plots + "\\BU_balances")
    
    
    # load plots
    plot_load_charts(file, param, nodes, dir_plots + "\\load_charts")
    
    
    # Plot capacities and generation
    plot_capacities(file, param, nodes, dir_plots + "\\capacity_plots")
    
    # Plot BU storage SOCs
    plot_storage_soc(file, param, dir_plots + "\\soc_storages")





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
    
    

def plot_capacities(file, param, nodes, dir_caps):
    
    print("Creating capacity plots...")
    
    dev_colors = get_compColor()
    time_steps = range(24)
    n_days = param["n_clusters"]
    
    if not os.path.exists(dir_caps):
        os.makedirs(dir_caps)     
    
    # Plot buildings devices
    all_devs_dom = ["HP", "EH","BOI", "CC", "free_cooler", "air_cooler"]
    all_flows = ["heat_HP", "heat_EH", "heat_BOI", "cool_CC", "cool_free_cooler", "cool_air_cooler"]
    
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
        plt.xticks(ind, ["HP", "EH", "BOI", "CC", "FRC", "AIR"])
        plt.legend(plot, ("HP", "EH", "BOI", "CC", "FRC", "AIR"), loc="upper center",ncol=3, fontsize=7)
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
        plt.xticks(ind, ["HP", "EH", "BOI", "CC", "FRC", "AIR"])   
        y_max = ax.get_ylim()[1]
        ax.set_ylim(top=1.25*y_max)
        
        fig.savefig(fname = dir_caps + "\\" +str(n)+".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
        plt.close(fig)
        
        
    # plot BU capacities
    all_devs = ["BOI", "CHP", "HP", "EH", "CC", "AC", "air_cooler"]
    all_flows = ["heat_BOI", "power_CHP", "heat_HP", "heat_EH", "cool_CC", "cool_AC", "cool_air_cooler"]
    
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
            res_thermal = read_energy_flow(file, "residual_thermal_demand", n, param)
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
                 "cool_free_cooler", 
                 "cool_air_cooler",
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
            ax.legend(loc='lower center', ncol=3) 
            
            # Cooling balance
            ax = fig.add_subplot(2,1,2, ylabel = "Cool [kW]")         
            # Sources
            cool_sources = ["cool_CC", "cool_free_cooler", "cool_air_cooler"]
            plot_series = np.zeros((len(cool_sources), len(time_steps)))
            for k in range(len(cool_sources)):
                plot_series[k,:] = series[cool_sources[k]][d]
            plot_colors = tuple(flow_colors[flow] for flow in cool_sources)
            ax.stackplot(time_steps, plot_series, step="pre", labels = cool_sources, colors = plot_colors, zorder = -100)            
            # sinks
            ax.step(time_steps, series["cool_dem"][d], label = "cool_dem", color = "k", linewidth = 2, zorder = -1)
            ax.legend(loc = "lower center", ncol= 2) 
                        
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
                 "cool_air_cooler",
                 "power_from_grid", "power_to_grid",
                 "power_PV"
                 ]
    
    # list of all demands
#    all_dem = ["heat",
#               "cool",
#               "power"
#               ]
                 
        
    
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
    
    
    # Create plots for every building
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
        cool_sources = ["cool_CC", "cool_AC", "cool_air_cooler", "dch_CTES"]
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
        power_sources = ["power_CHP", "power_PV", "power_from_grid"]
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
        ax.step(time_steps, series["power_dem"][d] + series["power_HP"][d] + series["power_CC"][d] + series["power_EH"][d] + series["power_to_grid"][d], label = "power_to_grid", color = flow_colors["power_to_grid"], linewidth = 2, zorder = -40)
        ax.set_ylim(bottom = 0, top= ax.get_ylim()[1]*1.3)
        ax.legend(loc = "upper center", ncol= 4, fontsize = 5)         
        
        
        fig.subplots_adjust(hspace = 0.2)
        fig.savefig(fname = dir_plots + "\Day " + str(d) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
#            plt.show()
        plt.close(fig)
#                
         
    print("All plots created!")
                
                

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





#def read_solution_file(file_name, param, nodes):
#    
#    start_time = time.time()
#    
#    time_steps = range(24)
#    
#    # Number of type-days
#    n_days = param["n_clusters"]
#    
#    # Read file
#    with open(file_name, "r") as solution_file:
#        file = solution_file.readlines()
#    
#    
#    # Read time series of energy flows of all devices
#    
#    # Prepare dictionary
#    energy_flows = {}    
#    for system in ["buildings", "balancing_unit"]:
#        energy_flows[system] = {}                        
#        for flow in ["heat", "cool", "power"]:
#            energy_flows[system][flow] = {}
#            for function in ["sources", "sinks"]:
#                energy_flows[system][flow][function] = {}
#                if system == "buildings":
#                    for n in nodes:
#                        energy_flows[system][flow][function][n] = {}
#
#    
#    # Read building energy flows
#    for n in nodes:
#
#        # heat sources
#        for flow in ["heat_HP", "heat_EH", "heat_BOI"]:                        
#            energy_flows["buildings"]["heat"]["sources"][n][flow] = read_energy_flows(flow, file, n_days, n)
#        for flow in ["dch_TES"]:
#            energy_flows["buildings"]["heat"]["sources"][n][flow] = read_energy_flows(flow, file, n_days, n)
#            
#        # heat sinks
#        for flow in ["heat_dem"]:
#            energy_flows["buildings"]["heat"]["sinks"][n][flow] = nodes[n]["heat"]
#        for flow in ["ch_TES"]:
#            energy_flows["buildings"]["heat"]["sinks"][n][flow] = read_energy_flows(flow, file, n_days, n)
#            
#        # cooling sources
#        for flow in ["cool_CC", "cool_free_cooler", "cool_air_cooler"]:
#            energy_flows["buildings"]["cool"]["sources"][n][flow] = read_energy_flows(flow, file, n_days, n)
#        
#        # cooling sinks
#        for flow in ["cool_dem"]:
#            energy_flows["buildings"]["cool"]["sinks"][n][flow] = nodes[n]["cool"]
#            
#        # power sources
#        for flow in ["power_PV"]:
#            energy_flows["buildings"]["power"]["sources"][n][flow] = read_energy_flows(flow, file, n_days, n)
#        #BATTERIE HIER FORMAL MITSCHLEPPEN
#            
#        # power sinks
#        for flow in ["power_HP", "power_CC", "power_EH"]:
#            energy_flows["buildings"]["power"]["sinks"][n][flow] = read_energy_flows(flow, file, n_days, n)
#            
#    
#    
#    # Read balancing unit energy flows
#    
#    # Get residual heating and cooling demand
#    # !!!!    
#
#    # heat sources
#    for device in ["heat_BOI", "heat_CHP", "heat_HP", "heat_EH"]:
#        energy_flows["balancing_unit"]["heat"]["sources"][device] = read_energy_flows(flow, file, n_days)
#    for device in ["dch_TES"]:
#        energy_flows["balancing_unit"]["heat"]["sources"][device] = read_energy_flows(flow, file, n_days)
#        
#    # heat sinks
##    for device in ["residual"]:
##        energy_flows["balancing_unit"]["heat"]["sinks"][device] = 
#            
#     
#    return energy_flows






#energy_flows = read_solution_file(file_name, param, nodes)






    
    
def get_compColor():
    """
    This function defines a color for each device that is used for plots.
    
    """
    
    compColor = {}
    
#    compColor["power_supply"] = "k"
#    compColor["heat_dem"] = "k"
#    compColor["cool_dem"] = "k"
    
    compColor["BOI"] = (0.843, 0.059, 0.059, 0.8)
    compColor["heat_BOI"] = (0.843, 0.059, 0.059, 0.8)
#    compColor["BOI_h"] = compColor["BOI"]
    compColor["CHP"] = (0.137, 0.706, 0.196, 0.8)
    compColor["heat_CHP"] = (0.137, 0.706, 0.196, 0.8)
    compColor["power_CHP"] = (0.137, 0.706, 0.196, 0.8)
#    compColor["CHP_ICE_h"] = compColor["CHP_ICE"]             
#    compColor["CHP_ICE_p"] = compColor["CHP_ICE"]
#    compColor["CHP_GT"] = (0.667, 0.824, 0.549, 0.6)
#    compColor["CHP_GT_h"] = compColor["CHP_GT"]             
#    compColor["CHP_GT_p"] = compColor["CHP_GT"]
    compColor["EH"] = (0.961, 0.412, 0.412, 0.8)
    compColor["heat_EH"] = (0.961, 0.412, 0.412, 0.8)
    compColor["power_EH"] = (0.961, 0.412, 0.412, 0.8)
#    compColor["heat_EH"] = compColor["EH"]       
#    compColor["EH_p"] = compColor["EH"]
    compColor["HP"] = (0.471, 0.843, 1.0, 0.8)
    compColor["heat_HP"] = compColor["HP"]
    compColor["power_HP"] = compColor["HP"]
#    compColor["heat_HP"] = compColor["HP"]
#    compColor["heat_consum_HP"] = compColor["HP"]

#    compColor["HP_aw_p"] = compColor["HP_aw"]
#    compColor["HP_ww"] = (0.471, 0.843, 1.0, 0.8)
#    compColor["HP_ww_h"] = compColor["HP_ww"]  
#    compColor["HP_ww_p"] = compColor["HP_ww"]
    compColor["PV"] = (1.000, 0.725, 0.000, 0.8)
    compColor["power_PV"] = (1.000, 0.725, 0.000, 0.8)
#    compColor["PV_curtail"] = (1.000, 0.725, 0.000, 0.3)
#    compColor["PV_fac"] = compColor["PV"]
#    compColor["STC"] = (0.922, 0.471, 0.039, 0.8)
#    compColor["STC_curtail"] = (0.922, 0.471, 0.039, 0.3)
#    compColor["WT"] = (0.098, 0.843, 0.588, 0.8)
#    compColor["WT_curtail"] = (0.098, 0.843, 0.588, 0.3)
    compColor["AC"] = (0.529, 0.706, 0.882, 0.8)
    compColor["cool_AC"] = (0.529, 0.706, 0.882, 0.8)
    compColor["heat_AC"] = (0.529, 0.706, 0.882, 0.8)
#    compColor["AC_h"] = compColor["AC"]
#    compColor["AC_c"] = compColor["AC"]
    compColor["CC"] = (0.184, 0.459, 0.710, 0.8)
    compColor["cool_CC"] = (0.184, 0.459, 0.710, 0.8)
    compColor["power_CC"] = (0.184, 0.459, 0.710, 0.8)
#    compColor["cool_CC"] = compColor["CC"]      
#    compColor["CC_p"] = compColor["CC"]
#    compColor["BAT"] = (0.482, 0.482, 0.482, 0.8)
#    compColor["BAT_ch"] = compColor["BAT"] 
#    compColor["BAT_dch"] = compColor["BAT"]
    compColor["TES"] = (0.482, 0.482, 0.482, 0.8)
    compColor["ch_TES"] = (0.482, 0.482, 0.482, 0.8)
    compColor["dch_TES"] = (0.482, 0.482, 0.482, 0.8)
#    compColor["TES_ch"] = compColor["TES"]           
#    compColor["TES_dch"] = compColor["TES"]
    compColor["CTES"] = (0.582, 0.582, 0.582, 0.8)
    compColor["ch_CTES"] = compColor["CTES"]
    compColor["dch_CTES"] = compColor["CTES"]
    compColor["free_cooler"] = compColor["AC"]
    compColor["cool_free_cooler"] = compColor["AC"]
    compColor["air_cooler"] = compColor["CTES"]
    compColor["cool_air_cooler"] = (0.282, 0.282, 0.282, 0.8)
#    compColor["ITES_ch"] = compColor["ITES"] 
#    compColor["ITES_dch"] = compColor["ITES"]
#    compColor["H2_TANK"] = (0.482, 0.482, 0.482, 0.8)
#    compColor["H2_TANK_ch"] = compColor["H2_TANK"] 
#    compColor["H2_TANK_dch"] = compColor["H2_TANK"]
    compColor["power_from_grid"] = (0.749, 0.749, 0.749, 1)
    compColor["power_to_grid"] = (0.749, 0.749, 0.749, 1)
#    compColor["GEN"] = (0.02, 0.627, 0.627, 0.8)
#    compColor["GEN_p"] = compColor["GEN"]
#    compColor["ELYZ"] = (0.706, 0.510, 0.843, 0.8)
#    compColor["ELYZ_p"] = compColor["ELYZ"]
#    compColor["FC"] = (0.549, 0.294, 0.784, 0.8)
#    compColor["FC_h"] = compColor["FC"]
#    compColor["FC_p"] = compColor["FC"]
#    compColor["CONV"] = (0.749, 0.749, 0.749, 1)
    
    return compColor
