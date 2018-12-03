# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 12:21:16 2018

@author: mwi
"""


import matplotlib.pyplot as plt
import os
import numpy as np

def print_building_devs(nodes, time_steps):
    
    sum_residual_heat = sum(nodes[n]["res_heat_dem"] for n in range(len(nodes)))
    print("Sum of residual heat: " + str(sum_residual_heat) + " kWh")
      
    for n in range(len(nodes)):
        sum_hp_power = sum(nodes[n]["power_HP"][t] for t in time_steps)
        sum_eh_power = sum(nodes[n]["power_EH"][t] for t in time_steps)
        sum_cc_power = sum(nodes[n]["power_CC"][t] for t in time_steps)
        
        print("Electric power of heat pump: " + str(sum_hp_power) + " kWh (" + nodes[n]["name"] + ")")
        print("Electric power of electric heater: " + str(sum_eh_power) + " kWh (" + nodes[n]["name"] + ")")
        print("Electric power of chiller: " + str(sum_cc_power) + " kWh (" + nodes[n]["name"] + ")")

    sum_power = sum(nodes[n]["power_HP"] + nodes[n]["power_EH"] + nodes[n]["power_CC"] for n in range(len(nodes)))  
    print("Electric power: " + str(np.sum(sum_power)) + " kWh")

    total_heat_residual = sum(i for i in sum_residual_heat.tolist() if i > 0)
    total_cool_residual = sum(i for i in sum_residual_heat.tolist() if i < 0)

    print("Total heat residual: " + str(total_heat_residual) + " kWh")
    print("Total cooling residual: " + str(total_cool_residual) + " kWh")
    
def plot_residual_heat(dir_results):
    sum_residual_heat = np.loadtxt(open(dir_results + "\\sum_residual_heat.txt", "rb"), delimiter=",", usecols=(0))
    
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14})
    plt.plot(sum_residual_heat, color="red")
    plt.xlim(0, 8760)
    plt.tight_layout(h_pad=6)
    plt.ylabel("Residual heat inter-balancing [kW]")
    plt.xlabel("Time [hours]")
    plt.savefig(dir_results + "\\sum_residual_heat.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()  
    print("Created 'sum_residual_heat.png' sucessfully.")

def plot_total_power_dem_bldgs(dir_results):
    sum_power_dem_bldgs = np.loadtxt(open(dir_results + "\\sum_power_dem_bldgs.txt", "rb"), delimiter=",", usecols=(0))
    
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14})
    plt.plot(sum_power_dem_bldgs, color="green")
    plt.xlim(0, 8760)
    plt.tight_layout(h_pad=6)
    plt.ylabel("Total power demand (CC, HP, EH) [kW]")
    plt.xlabel("Time [hours]")
    plt.savefig(dir_results + "\\sum_power_dem_bldgs.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()  
    print("Created 'sum_power_dem_bldgs.png' sucessfully.")
    
def plot_power_dem_HP_EH_CC(dir_results):
    sum_power_dem_HP = np.loadtxt(open(dir_results + "\\sum_power_dem_HP.txt", "rb"), delimiter=",", usecols=(0))
    sum_power_dem_EH = np.loadtxt(open(dir_results + "\\sum_power_dem_EH.txt", "rb"), delimiter=",", usecols=(0))
    sum_power_dem_CC = np.loadtxt(open(dir_results + "\\sum_power_dem_CC.txt", "rb"), delimiter=",", usecols=(0))
    
    # Create figure
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14})
    plt.plot(sum_power_dem_HP, color="green")
    plt.plot(sum_power_dem_EH, color="red")
    plt.plot(sum_power_dem_CC, color="blue")
    plt.xlim(0, 8760)
    plt.tight_layout(h_pad=6)
    plt.ylabel("Total power demand (CC, HP, EH) [kW]")
    plt.xlabel("Time [hours]")
    plt.savefig(dir_results + "\\power_dem_HP_EH_CC.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()  
    print("Created 'power_dem_HP_EH_CC.png' sucessfully.")
    
def plot_demands(nodes, dir_results):
    for n in range(len(nodes)):
        # Create figure
        plt.figure(figsize=(12, 9))
        plt.rcParams.update({'font.size': 14})
        plt.plot(nodes[n]["heat"], color="red")
        plt.plot(nodes[n]["cool"], color="blue")
        plt.xlim(0, 8760)
        plt.tight_layout(h_pad=6)
        plt.ylabel("Total heating/cooling demand [kW]")
        plt.xlabel("Time [hours]")
        plt.savefig(dir_results + "\\heat_cool_dem_bldg_" + str(n) + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
        plt.clf()
        plt.close()  
        print("Created 'heat_cool_dem_bldg_" + str(n) + ".png' sucessfully.")
        
def plot_COP_HP_CC(param, dir_results):
    # Create figure
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14})
    plt.plot(param["COP_HP"], color="green")
    plt.plot(param["COP_CC"], color="blue")
    plt.xlim(0, 8760)
    plt.tight_layout(h_pad=6)
    plt.ylabel("COP heat pump and compression chiller [---]")
    plt.xlabel("Time [hours]")
    plt.savefig(dir_results + "\\COP_HP_CC.png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    plt.clf()
    plt.close()  
    print("Created 'COP_HP_CC.png' sucessfully.")

def calc_diversity_index(nodes, time_steps):
    
    sum_heat_dem = sum(nodes[n]["heat"] for n in range(len(nodes)))  
    sum_cool_dem = sum(nodes[n]["cool"] for n in range(len(nodes)))
    
    div_index_sum = 0
    counts = 0
    for t in time_steps:
        if sum_cool_dem[t]+sum_heat_dem[t] != 0:
            div_index_sum += (2 * (1-((sum_cool_dem[t]**2+sum_heat_dem[t]**2)/((sum_cool_dem[t]+sum_heat_dem[t])**2))))
            counts += 1
            
    div_index = div_index_sum / counts
    print("Diversity index: " + str(div_index))

def plot_ordered_load_curve(heat_dem_sort, hp_capacity, eh_capacity, param, bldg_id, time_steps, dir_results):
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14})
        
    plt.plot(heat_dem_sort)
    plt.plot([0, param["hours_el_heater_per_year"], param["hours_el_heater_per_year"]],[hp_capacity, hp_capacity, 0]) # [x1,x2],[y1,y2]
    plt.tight_layout(h_pad=6)
    
    plt.savefig(fname =  dir_results + "//bldg_balancing//Ordered_load_curve_" + str(bldg_id) +".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    print("Plot created: Ordered_load_curve_" + bldg_id +".png")
    plt.clf()
    plt.close()
    
def plot_bldg_balancing(node, time_steps, param, dir_results):

    # start/end day of each month
    startD = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    endD = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
#    MonthTuple = ("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")

    h = {} # contains all 8760 time steps
    H = {} # help dicitionary
    d = {} # contains mean values for 365 days
    D = {} # help dicitionary
    M = {} # contains mean values for 12 months

    dev_list = ["res_heat_dem", "cool_dem", "heat_dem", "power_dem", "heat_HP", "heat_EH", "heat_consum_HP", "cool_CC"]  
               
    for device in dev_list:
        h[device] = np.zeros(8760)
        H[device] = {}
        d[device] = np.zeros(365)
        D[device] = {}
        M[device] = np.zeros(12)
    
#    h["power_dem"] = node["power"]    
    h["heat_dem"] = node["heat"]    
    h["cool_dem"] = node["cool"]
    h["res_heat_dem"] = node["res_heat_dem"]
    h["power_HP"] = node["power_HP"]
    h["power_EH"] = node["power_EH"]
    h["power_CC"] = node["power_CC"]
    h["heat_HP"] = node["power_HP"] * param["COP_HP"]
    h["heat_consum_HP"] = (-1) * (node["power_HP"] * param["COP_HP"] - node["power_HP"])
    
    h["cool_CC"] = node["power_CC"] * param["COP_CC"]
    h["heat_EH"] = node["power_EH"] / param["eta_th_eh"]
            
    # divide time series (8760 time steps) into daily (H), monthly (D) and yearly profiles (M)
    for device in dev_list:
        for k in range(365):
            H[device][k] = h[device][(k*24):(k*24+24)]
            d[device][k] = sum(h[device][(k*24):(k*24+24)])/24
    
        for m in range(12):
            D[device][m] = d[device][startD[m]:endD[m]]
            M[device][m] = sum(d[device][startD[m]:endD[m]])/(endD[m]-startD[m])  
            
    dev_list_heat = ["heat_dem", "heat_HP", "heat_EH"]
    dev_list_resi = ["res_heat_dem", "heat_consum_HP", "cool_CC"]  
    dev_list_cool = ["cool_dem", "cool_CC"]
    heat_dict = {}
    resi_dict = {}
    cool_dict = {}

    #%% Plot yearly profile
#    if plot_mode["yearly"] == 1:
        
    for dev in dev_list_heat:
        heat_dict[dev] = M[dev]
    for dev in dev_list_resi:
        resi_dict[dev] = M[dev]  
    for dev in dev_list_cool:
        cool_dict[dev] = M[dev]            
    save_name = dir_results + "//bldg_balancing//Year_Profile"
    dir_results + "//bldg_balancing//Year_Balancing_" + str(node["name"]) +".png"
    plot_interval(heat_dict, resi_dict, cool_dict, save_name, "Month")

    
    #%% plots a time series of abitrary length
def plot_interval(heat, resi, cool, save_name, xTitle):
    compColor = get_compColor()

    # Transform data for plots: heating
    heat_res = {}    
    heat_labels = []
    heat_res_list = []
    heat_color = []
    for device in ["heat_HP", "heat_EH"]:
        heat_res[device] = np.zeros(2*heat[device].size)
        for t in range(heat[device].size):            
            heat_res[device][2*t:2*t+2] = [heat[device][t], heat[device][t]]
        heat_res_list.extend([heat_res[device].tolist()])
        heat_labels.extend([device])
        heat_color.extend([compColor[device]])    
    for device in ["heat_dem"]:    
        heat_res[device] = np.zeros(2*heat[device].size)
        for t in range(heat[device].size):
            heat_res[device][2*t:2*t+2] = [heat[device][t], heat[device][t]]
            
    # Transform data for plots: residual heat
    resi_res = {}    
    resi_labels = []
    resi_res_list = []
    resi_color = []
    for device in ["heat_consum_HP", "cool_CC"]:
        resi_res[device] = np.zeros(2*resi[device].size)
        for t in range(resi[device].size):            
            resi_res[device][2*t:2*t+2] = [resi[device][t], resi[device][t]]
        resi_res_list.extend([resi_res[device].tolist()])
        resi_labels.extend([device])
        resi_color.extend([compColor[device]])
    for device in ["res_heat_dem"]:    
        resi_res[device] = np.zeros(2*resi[device].size)
        for t in range(resi[device].size):
            resi_res[device][2*t:2*t+2] = [resi[device][t], resi[device][t]]
            
    # Transform data for plots: cooling
    cool_res = {}    
    cool_labels = []
    cool_res_list = []
    cool_color = []
    for device in ["cool_CC"]:
        cool_res[device] = np.zeros(2*cool[device].size)
        for t in range(cool[device].size):            
            cool_res[device][2*t:2*t+2] = [cool[device][t], cool[device][t]]
        cool_res_list.extend([cool_res[device].tolist()])
        cool_labels.extend([device])
        cool_color.extend([compColor[device]])    
    for device in ["cool_dem"]:    
        cool_res[device] = np.zeros(2*cool[device].size)
        for t in range(cool[device].size):
            cool_res[device][2*t:2*t+2] = [cool[device][t], cool[device][t]]
    

    # Create time ticks for x-axis
    timeTicks = [0]
    for t in range(heat["heat_dem"].size):
        timeTicks.extend([timeTicks[-1] + 1])
        timeTicks.extend([timeTicks[-2] + 1])
    del timeTicks[-1]

    # Create figure
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14})

    # Create subplot: heat balance
    plt.subplot(311, ylabel = 'Heating output in kW', xlabel = " ")
    plt.stackplot(timeTicks, np.vstack(heat_res_list), labels=heat_labels, colors=heat_color)
    plt.plot(timeTicks, heat_res["heat_dem"], color="black", linewidth = 3, label="Heating demand")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(min(timeTicks), max(timeTicks)+1, 1))
    plt.xlim(min(timeTicks), max(timeTicks))
    plt.tight_layout(h_pad=6)
    
    # Create second subplot: power balance
    plt.subplot(312, ylabel = "Heat balance in kW", xlabel = xTitle)
    plt.stackplot(timeTicks, np.vstack(resi_res_list), labels=resi_labels, colors=resi_color)
    plt.plot(timeTicks, resi_res["res_heat_dem"], color='black', linewidth = 3, label="Power demand")


    # Create third subplot: cooling balance
    plt.subplot(313, ylabel = 'Cooling output in kW', xlabel = " ")
    plt.stackplot(timeTicks, np.vstack(cool_res_list), labels=cool_labels, colors=cool_color)
    plt.plot(timeTicks,cool_res["cool_dem"], color='black', linewidth = 3, label="Cooling demand")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(min(timeTicks), max(timeTicks)+1, 1))
    plt.xlim(min(timeTicks), max(timeTicks))


    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(min(timeTicks), max(timeTicks)+1, 1))
    plt.xlim(min(timeTicks), max(timeTicks))

    plt.tight_layout(h_pad=6)
    plt.savefig(fname = save_name + ".png", dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1)
    print("Plot created: " + save_name + ".png")
    plt.clf()
    plt.close()
    
    
def save_balancing_results(node, time_steps, dir_results):

    fo = open(dir_results + "//bldg_balancing//" + node["name"] + ".txt", "w")
    
    fo.write("Capacity heat pump: " + str(node["hp_capacity"]) + "\n")
    fo.write("Capacity electric heater: " + str(node["eh_capacity"]) + "\n")
    fo.write("Capacity compression chiller: " + str(node["cc_capacity"]) + "\n")
    
    
    fo.write("\nHeating demand:\n")
    for t in time_steps:
        fo.write(str(node["heat"][t]) + "\n")
        
    fo.write("\nCooling demand:\n")
    for t in time_steps:
        fo.write(str(node["cool"][t]) + "\n")
    
    fo.write("\nPower heat pump:\n")
    for t in time_steps:
        fo.write(str(node["power_HP"][t]) + "\n")
        
    fo.write("\nPower electric heater:\n")
    for t in time_steps:
        fo.write(str(node["power_EH"][t]) + "\n")

    fo.write("\nPower compression chiller:\n")
    for t in time_steps:
        fo.write(str(node["power_CC"][t]) + "\n")
        
    fo.write("\nBalancing heat (> 0: mass flow from supply to return):\n")
    for t in time_steps:
        fo.write(str(node["res_heat_dem"][t]) + "\n")    
        
    
    
#    fo.write("\nElectricity demand:\n")
#    for t in time_steps:
#        fo.write(str(node["power"][t]) + str("\n"))
    
def create_result_folders(dir_results):
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
            
    if not os.path.exists(dir_results + "//bldg_balancing//"):
        os.makedirs(dir_results + "//bldg_balancing//")
        
    if not os.path.exists(dir_results + "//balancing_unit//"):
        os.makedirs(dir_results + "//balancing_unit//")
        
    if not os.path.exists(dir_results + "//topology//"):
        os.makedirs(dir_results + "//topology//")
        
        
def plot_network(nodes, cap):
    import networkx as nx

#    # Define plot area
    rel_frame = 0.1 # distance between nodes and image frame
    lim = {"x_max": nodes[0]["x"],
           "x_min": nodes[0]["x"],
           "y_max": nodes[0]["y"],
           "y_min": nodes[0]["y"],
           }

    for n in range(len(nodes)):
        if nodes[n]["x"] > lim["x_max"]:
            lim["x_max"] = nodes[n]["x"]
        if nodes[n]["y"] > lim["y_max"]:
            lim["y_max"] = nodes[n]["y"]
        if nodes[n]["x"] < lim["x_min"]:
            lim["x_min"] = nodes[n]["x"]
        if nodes[n]["y"] < lim["y_min"]:
            lim["y_min"] = nodes[n]["y"]
    
    x_lim = [lim["x_min"] - rel_frame*(lim["x_max"]-lim["x_min"]), lim["x_max"] + rel_frame*(lim["x_max"]-lim["x_min"])] 
    y_lim = [lim["y_min"] - rel_frame*(lim["y_max"]-lim["y_min"]), lim["y_max"] + rel_frame*(lim["y_max"]-lim["y_min"])]
  
    graph_pipes = nx.Graph()
    for k in node_list:
        graph_pipes.add_node(k,pos=(nodes[k]["x"], nodes[k]["y"]))

#    list_weighted_edges = []
#    for e in edges:
#        list_weighted_edges.append((edge_dict_rev[e][0], edge_dict_rev[e][1], cap[e].X))
#            
#    graph_pipes.add_weighted_edges_from(list_weighted_edges)
#    
#    # Plot pipe installation
#    pos = nx.get_node_attributes(graph_pipes, "pos")
#    weights_pipe = [graph_pipes[u][v]["weight"] for u,v in graph_pipes.edges()]
#    fig, ax = plt.subplots(figsize=(10,5))
#    nx.draw(graph_pipes, pos, with_labels=True, font_weight="bold", width=weights_pipe)
#    nx.draw_networkx_nodes(graph_pipes,pos,nodelist=[balancing_node],node_color="blue") # Highlight balancing unit
#    plt.axis("equal")
#    plt.ylim(y_lim)
#    plt.xlim(x_lim)
#    file_name = "Pipe_installation.png"
#    plt.savefig(file_name, dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1, transparent=True)
#    print("Created plot '" + file_name + "' successfully.")
#    plt.clf()
#    plt.close()
#    
#    for t in time_steps:
#        
#        graph_flow = nx.Graph()
#        for k in node_list:
#            graph_flow.add_node(k,pos=(nodes[k]["x"], nodes[k]["y"]))
#        list_weighted_edges = []
#        for e in edges:
#            list_weighted_edges.append((edge_dict_rev[e][0], edge_dict_rev[e][1], m_dot[e,t].X))
#        graph_flow.add_weighted_edges_from(list_weighted_edges)
#        weights_flow = [graph_flow[u][v]["weight"] for u,v in graph_flow.edges()]
#        fig, ax = plt.subplots(figsize=(10,5))
#        nx.draw(graph_pipes, pos, with_labels=True, font_weight="bold", width=weights_pipe)
#        nx.draw_networkx_nodes(graph_pipes,pos,nodelist=[balancing_node],node_color="blue") # Highlight balancing unit
#        nx.draw(graph_flow, pos, font_weight="bold", width=weights_flow, edge_color="red")
#        plt.axis("equal")
#        plt.ylim(y_lim)
#        plt.xlim(x_lim)
#        file_name = "Mass_flow_t" + str(t) + ".png"
#        plt.savefig(file_name, dpi = 200, format = "png", bbox_inches="tight", pad_inches=0.1, transparent=True)
#        print("Created plot '" + file_name + "' successfully.")
#        plt.clf()
#        plt.close()
#    
#    nx.write_gml(graph_pipes, "topology//graph_pipes.gml")          
              
def save_network_data(nodes, cap):
    pass
    #%% Print results
#    for node in node_list:
#        if ksi[node].X == 1:
#            balancing_node = node
#            print("Balancing unit installed at node: " + str(node))
#            for t in time_steps:
#                print("Balancing power: t" + str(t) + ": " + str(round(m_bal[node,t].X,2)))
#            
#    print("Pipe capacities:")
#    for edge in edges:
#        print("Edge [" + str(edge_dict_rev[edge][0]) + "-" + str(edge_dict_rev[edge][1]) + "]" + ": " + str(round(cap[edge].X,2)) + " [x: " + str(x[edge].X) + "]")
#        
#    print("\nMass flows capacities:")    
#    for t in time_steps:
#        for edge in edges:
#            print("m_dot" + str(edge) + "_t" + str(t) + ": " + str(round(m_dot[edge,t].X,2)))
#        print("\n")
    
    
def get_compColor():
    """
    This function defines a color for each device that is used for plots.
    
    """
    
    compColor = {}
#    compColor["BOI"] = (0.843, 0.059, 0.059, 0.8)
#    compColor["BOI_h"] = compColor["BOI"]
#    compColor["CHP_ICE"] = (0.137, 0.706, 0.196, 0.8)
#    compColor["CHP_ICE_h"] = compColor["CHP_ICE"]             
#    compColor["CHP_ICE_p"] = compColor["CHP_ICE"]
#    compColor["CHP_GT"] = (0.667, 0.824, 0.549, 0.6)
#    compColor["CHP_GT_h"] = compColor["CHP_GT"]             
#    compColor["CHP_GT_p"] = compColor["CHP_GT"]
    compColor["EH"] = (0.961, 0.412, 0.412, 0.8)
    compColor["heat_EH"] = compColor["EH"]       
#    compColor["EH_p"] = compColor["EH"]
    compColor["HP"] = (0.0, 0.706, 0.804, 0.8)
    compColor["heat_HP"] = compColor["HP"]
    compColor["heat_consum_HP"] = compColor["HP"]

#    compColor["HP_aw_p"] = compColor["HP_aw"]
#    compColor["HP_ww"] = (0.471, 0.843, 1.0, 0.8)
#    compColor["HP_ww_h"] = compColor["HP_ww"]  
#    compColor["HP_ww_p"] = compColor["HP_ww"]
#    compColor["PV"] = (1.000, 0.725, 0.000, 0.8)
#    compColor["PV_curtail"] = (1.000, 0.725, 0.000, 0.3)
#    compColor["PV_fac"] = compColor["PV"]
#    compColor["STC"] = (0.922, 0.471, 0.039, 0.8)
#    compColor["STC_curtail"] = (0.922, 0.471, 0.039, 0.3)
#    compColor["WT"] = (0.098, 0.843, 0.588, 0.8)
#    compColor["WT_curtail"] = (0.098, 0.843, 0.588, 0.3)
#    compColor["AC"] = (0.529, 0.706, 0.882, 0.8)
#    compColor["AC_h"] = compColor["AC"]
#    compColor["AC_c"] = compColor["AC"]
    compColor["CC"] = (0.184, 0.459, 0.710, 0.8)
    compColor["cool_CC"] = compColor["CC"]      
#    compColor["CC_p"] = compColor["CC"]
#    compColor["BAT"] = (0.482, 0.482, 0.482, 0.8)
#    compColor["BAT_ch"] = compColor["BAT"] 
#    compColor["BAT_dch"] = compColor["BAT"]
#    compColor["TES"] = (0.482, 0.482, 0.482, 0.8)
#    compColor["TES_ch"] = compColor["TES"]           
#    compColor["TES_dch"] = compColor["TES"]
#    compColor["ITES"] = (0.482, 0.482, 0.482, 0.8)
#    compColor["ITES_ch"] = compColor["ITES"] 
#    compColor["ITES_dch"] = compColor["ITES"]
#    compColor["H2_TANK"] = (0.482, 0.482, 0.482, 0.8)
#    compColor["H2_TANK_ch"] = compColor["H2_TANK"] 
#    compColor["H2_TANK_dch"] = compColor["H2_TANK"]
#    compColor["power_from_grid"] = (0.749, 0.749, 0.749, 1)
#    compColor["power_to_grid"] = (0.749, 0.749, 0.749, 1)
#    compColor["GEN"] = (0.02, 0.627, 0.627, 0.8)
#    compColor["GEN_p"] = compColor["GEN"]
#    compColor["ELYZ"] = (0.706, 0.510, 0.843, 0.8)
#    compColor["ELYZ_p"] = compColor["ELYZ"]
#    compColor["FC"] = (0.549, 0.294, 0.784, 0.8)
#    compColor["FC_h"] = compColor["FC"]
#    compColor["FC_p"] = compColor["FC"]
#    compColor["CONV"] = (0.749, 0.749, 0.749, 1)
    
    return compColor
