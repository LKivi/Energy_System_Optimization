# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:28:12 2018

@author: mwi
"""

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import parameters as par
import post_processing
import os
import datetime
import time
import numpy as np
import json
import soil


def run_optim(nodes, param, dir_results):
    
    
    # Get edge and node data
    edge_dict, edge_dict_rev, edges, compl_graph, edge_list, edge_lengths = par.get_edge_dict(len(nodes),nodes)
#    param = par.calc_pipe_costs(nodes, edges, edge_dict_rev, param)
    node_list = range(len(nodes))     
    
    dir_network = dir_results + "\\network"
    
    #%% STEP ONE: FIND NETWORK WITH MINIMUM TOTAL PIPE LENGTH (MINIMUM SPANNING TREE)
  
    dir_topo = dir_network + "\\topology"
    
    # Create networkx-graph
    weighted_graph = nx.Graph()
    
    nbunch = compl_graph.nodes
    weighted_graph.add_nodes_from(nbunch)
    
    ebunch = compl_graph.edges()
    ebunch_weighted = []
    for e in ebunch:
        ebunch_weighted.append((e[0], e[1], edge_lengths[edge_dict[e]]))
    weighted_graph.add_weighted_edges_from(ebunch_weighted)
    
    
     # find network with minimal length   
    network = nx.minimum_spanning_tree(weighted_graph)

    # Close network to ring network
    print("Achtung!! Hier wird das Netzwerk manuell zu einem Ringnetzwerk geschlossen. Bei einem anderen Use-Case muss die Stelle angepasst werden!")
    network.add_edge(0,16)
     
    network_length = sum(edge_lengths[edge_dict[e]] for e in network.edges)
    print("Pipe connections calculated. Total network length: " + str(network_length) + " m.")
    
    
    # get node positions
    pos = {}
    for k in node_list:
        pos[k] = (nodes[k]["x"], nodes[k]["y"])
   

    # Plot Network
    nx.draw(network, pos, with_labels=True, font_weight="bold")

    plt.grid(True)
    plt.axis("equal")    
#    plt.show()   

    # Save figure
    if not os.path.exists(dir_topo):
        os.makedirs(dir_topo)
        
    plt.savefig(dir_topo + "\\graph_min_length.png")
    
    # delete plot out of cache
    plt.clf()
        

        
        

    #%% STEP TWO: PLACE BALANCING UNIT FOR MINIMUM PIPE MASS FLOWS   
    
    dir_BU = dir_network + "\\mass_flows"
    
    # list of time_steps
    time_steps = range(8760)
    
    # get network pipe ids
    pipes = list(edge_dict[e] for e in network.edges)
    
    # Create a new model
    model = gp.Model("Ectogrid_BU_placement")
  
             
    # Create Variables             
    
    m_dot = model.addVars(pipes, time_steps, vtype = "C", lb = -1000, name = "mass_flow")     # kg/s,     mass flow through placed pipes
    
    ksi = model.addVars(node_list, vtype="B", name="balancing_unit")                                # ---,      Binary decision: balancing unit installed in node
    
    m_bal = model.addVars(node_list, time_steps, vtype="C", lb=-1000, name="mass_flow_balancing")   # kg/s,     Mass flow from cold pipe to hot pipe through balancing unit
    
    cap = model.addVars(pipes, vtype = "C", name = "pipe_capacities")                               # kg/s,     pipe capacities
    
    obj = model.addVar(vtype = "C", name = "objective")                                             # kg/s,     objective function
    

    # Define objective function
    model.update()
    model.setObjective(obj, gp.GRB.MINIMIZE)
    
    
    # Constraints
    
    # objective: total sum of mass flows 
    model.addConstr(obj == sum(sum(m_dot[p,t] for p in pipes) for t in time_steps))
    
 
    # Node balance   
    for node in node_list:
        edges_minus = []
        edges_plus = []
        
        nodes_lower = np.arange(node)                       # nodes which have lower ID-number
        nodes_higher = np.arange(node+1, len(node_list))    # nodes which have higher ID-number
        
        for node_from in nodes_lower:
            edge_id = edge_dict[(node_from, node)]
            if edge_id in pipes:
                edges_plus.append(edge_id)
        for node_to in nodes_higher:
            edge_id = edge_dict[(node, node_to)]
            if edge_id in pipes:
                edges_minus.append(edge_id)
            
        for t in time_steps:
            model.addConstr( sum(m_dot[k,t] for k in edges_plus) - sum(m_dot[k,t] for k in edges_minus) + m_bal[node,t] == nodes[node]["mass_flow"][t] )       
    
 
    # Maximum number of balancing units
    model.addConstr(sum(ksi[node] for node in node_list) <= param["number_of_balancing_units"], "number_balancing_units")
    
    # Balancing power is only at balancing nodes possible
    for node in node_list:
        for t in time_steps:
            model.addConstr(m_bal[node,t] <= ksi[node] * 1000)
            model.addConstr(-m_bal[node,t] <= ksi[node] * 1000)
              
        
    # Mass flow on edge must not exceed pipe capacity   
    for p in pipes:
        for t in time_steps:
            model.addConstr(m_dot[p,t] <= cap[p])
            model.addConstr(-m_dot[p,t] <= cap[p])      
    
    
    # Execute calculation           
    model.optimize()
    
    if model.Status in (3,4) or model.SolCount == 0:  # "INFEASIBLE" or "INF_OR_UNBD"
        print("No feasible solution found; none of the nodes is a suitable location for balancing the network.")  
               
    # save results
    if not os.path.exists(dir_BU):
        os.makedirs(dir_BU)
     
    # Write files    
    model.write(dir_BU + "\\model.lp")
    model.write(dir_BU + "\\model.prm")
    model.write(dir_BU + "\\model.sol")
            
    print("Optimization done.\n")

    for node in node_list:
        if round(ksi[node].X,0) == 1:
            balancing_node = node
            print("Balancing unit installed at node: " + str(balancing_node))
            print("Max BU heating: " + str(max(m_bal[node,t].X for t in time_steps)) + " kg/s")
            print("Min BU heating: " + str(min(m_bal[node,t].X for t in time_steps)) + " kg/s")
     
    
    
    # Store mass flow information in dictionary
    dict_pipes = {}    
     
    # Print pipe capacities and store them
    dict_pipes["cap"] = {}
    print("Pipe caps:")
    for p in pipes:
        dict_pipes["cap"][p] = cap[p].X
        print("Pipe " + str(edge_dict_rev[p][0]) + "-" + str(edge_dict_rev[p][1]) + ": " + str(round(cap[p].X,2)) + " kg/s.")
    
    # store mass flows in every pipe at every time step
    dict_pipes["m_dot"] = {}
    for p in pipes:
        dict_pipes["m_dot"][p] = {}
        for t in time_steps:
            dict_pipes["m_dot"][p][t] = m_dot[p][t].X
    
    # Save dictionary as json file
    with open(dir_BU + "\mass_flows.json", "w") as outfile:
        json.dump(dict_pipes, outfile, indent=4, sort_keys=True)



#    # store maximum mass flows into each direction (maximum flow in one of the two directions equals pipe cap)
#    dict_pipes["m_max"] = {}
#    for p in pies:
#        dict_pipes["m_max"][p] = {} 
#        dict_pipes["m_max"][p]["plus"] = np.max(m_dot[p,t] for t in time_steps)
#        dict_pipes["m_max"][p]["minus"] = np.max(-m_dot[p,t] for t in time_steps)        
    
 
#    # Draw network and save plot
#    
#    graph = nx.Graph()
#    for k in node_list:
#        graph.add_node(k, pos=(nodes[k]["x"], nodes[k]["y"]))
#    
#    ebunch = []
#    for k in pipes:
#        ebunch.append((edge_dict_rev[k][0], edge_dict_rev[k][1], cap[k].X))
#    
#    graph.add_weighted_edges_from(ebunch)


    
#    # Plot pipe installation in black
#    pos = nx.get_node_attributes(graph, "pos")
#    weights = [graph[u][v]["weight"] for u,v in graph.edges()]
##    nx.draw(graph, pos, with_labels=True, font_weight="bold", width=weights)
        
    # Draw graph and highlight balancing unit
    
    nx.draw(network, pos, with_labels=True, font_weight="bold")
    nx.draw_networkx_nodes(network,pos,nodelist=[balancing_node],node_color="blue")
    
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    
    plt.savefig(dir_BU + "\\graph.png")
    
    
    
    
    
    
    #%% STEP THREE: PIPE DIAMETERS FOR MINIMAL ANNUAL COSTS
    
    
    dir_dia = dir_network + "\\diameters"
        
    # list of time_steps
    time_steps = range(8760)
    
    # Calculate minimum inner pipe diameters [m] due to limitation of pipe friction
    dict_pipes["d_min"] = {}
    for p in pipes:
        dict_pipes["d_min"][p] = ((8*dict_pipes["cap"][p]**2*param["f_fric"])/(param["rho_f"]*np.pi**2*param["dp_pipe"]))**0.2
        
    # pre-factor for pump power calculation
    prefac = 1000 * (8 * param["f_fric"])/(param["rho_f"]**2*np.pi**2*param["eta_pump"])
        
    # Maximum pipe diameter [m] (used for pump power linearization)
    d_max = param["d_pipe_max"]
    
    
    # Create a new model
    model = gp.Model("Ectogrid_pipe_diameters")
  
             
    # Create Variables 

    d = {}           
    for p in pipes:       
        d[p] = model.addVar(vtype = "C", name = "inner_diameter_p"+str(p))           # mm,  inner pipe diameters                       
    
    # pump caps [W] and pump power at each time step [W]
    pump_cap = {}
    pump_el = {}
    for p in pipes:
        pump_cap[p] = {}
        pump_el[p] = {}
        for line in ["plus", "minus"]:
            pump_cap[p][line] = model.addVar(vtype = "C", name = "pump_capacity_pipe_"+str(p)+"_"+line)                                           # kW,        pump capacities 
            pump_el[p][line] = {}
            for t in time_steps:
                pump_el[p][line][t] = model.addVar(vtype = "C", lb = -gp.GRB.INFINITY, name = "pump_power_pipe_"+str(p)+"_"+line+"_t"+str(t))     # kW,        pump power
              
    
    pump_energy_total = model.addVar(vtype="C", name="total_pump_energy")                               # MWh, total pump energy for one year
    
    tac_network = model.addVar(vtype = "C", name = "tac_network")                                       # kEUR/a,   total annualized costs for pipe system incl. pumping
    
    
    # Device investments
    inv = {}
    for dev in ["pipes", "pumps"]:
        inv[dev] = model.addVar(vtype="C", name="inv_"+dev)                                    # EUR, investment costs

    
    # Annualized pipe and pump device costs
    tac = {}
    for dev in ["pipes", "pumps"]:
        tac[dev] = model.addVar(vtype = "C", name="tac_"+dev)                                  # EUR/a,   annualized investment costs

    
    # Define objective function
    model.update()
    model.setObjective(tac_network, gp.GRB.MINIMIZE)
    
    
    # CONSTRAINTS
    
    #%% PIPES
    
    # minimum pipe diameter due to limitation of pressure gradient; maximum pipe diameter according to params
    for p in pipes:       
        model.addConstr(d[p] >= dict_pipes["d_min"][p]*1000)
        model.addConstr(d[p] <= d_max*1000)
    
    # pipe investment costs
    model.addConstr(inv["pipes"] >= sum((param["inv_earth_work"] + 2 * param["inv_pipe_PE"] * d[p] * d[p]) * edge_lengths[p] for p in pipes), name = "Q1")
#    model.addConstr(inv_pipes <= sum((0.5*param["inv_earth_work"] + param["inv_pipe_PE"] * d_in[p] * d_in[p]) * 2*edge_lengths[p] for p in pipes), name = "Q2")
    
    # annualized pipe costs
    model.addConstr(tac["pipes"] == inv["pipes"] * (param["ann_factor_pipe"] + param["cost_om_pipe"]))
    
    
    #%% PUMPS
    
    # pump power
    for p in pipes:
        d_min = dict_pipes["d_min"][p]              # m,   minimum pipe diameter according to limitation of pressure gradient
        for t in time_steps:
            m_dot = dict_pipes["m_dot"][p][t]       # mass flow through pipe p at time step t
            if m_dot >= 0:
                model.addConstr(pump_el[p]["plus"][t] == prefac * 2 * edge_lengths[p] * m_dot**3 * ( 1/d_min**5 - (1/d_min**5 - 1/d_max**5)*(d[p]/1000 - d_min)/(d_max - d_min)))    # kW,   linear interpolation of pump energy between d_min and d_max
                model.addConstr(pump_el[p]["minus"][t] == 0)
            else:
                model.addConstr(pump_el[p]["plus"][t] == 0)
                model.addConstr(pump_el[p]["minus"][t] == prefac * 2 * edge_lengths[p] * (-m_dot)**3 * ( 1/d_min**5 - (1/d_min**5 - 1/d_max**5)*(d[p]/1000 - d_min)/(d_max - d_min)))    # kW,  linear interpolation of pump energy between d_min and d_max

    # sum up total pump energy [MWh]
    model.addConstr(pump_energy_total == sum(sum(sum(pump_el[p][line][t] for t in time_steps) for line in ["plus", "minus"]) for p in pipes) / 1000 )                            
    
    
    # pump power <= pump capacity     
    for p in pipes:
        for t in time_steps:
            for line in ["plus", "minus"]:
                model.addConstr( pump_el[p][line][t] <= pump_cap[p][line] )
                
    
    # pump investment costs
    model.addConstr(inv["pumps"] == sum(sum(pump_cap[p][line] for line in ["plus", "minus"]) for p in pipes) * param["inv_pump"])
    
    # annualized pump costs
    model.addConstr(tac["pumps"] == inv["pumps"] * (param["ann_factor_pump"] + param["cost_om_pump"]))

        
    
    #%% OBJECTIVE
    
    model.addConstr( tac_network == (tac["pipes"]
                                    + tac["pump"]
                                    + pump_energy_total * param["price_el_pumps"]) / 1000 )          # kEUR/a,      total annual network costs
                                   
    
    
    # Execute calculation           
    model.optimize()
    
    # save results
    if not os.path.exists(dir_dia):
        os.makedirs(dir_dia)
        
    model.write(dir_dia + "\\model.lp")
    model.write(dir_dia + "\\model.prm")
    model.write(dir_dia + "\\model.sol")
            
    print("Optimization done.\n")
    
    # Print pipe diameters
    print("Pipe diameters:")
#    for p in pipes:
#        print("Pipe " + str(edge_dict_rev[p][0]) + "-" + str(edge_dict_rev[p][1]) + ": " + str(round(d_in[p].X*10,2)) + " mm.")    

    

    #%% CHOOSE NORM PIPE-DIAMETERS
       
    # Choose pipe diameter from list of available diameters
    pipes_norm = {}
        
    # available inner pipe diameters for the network    
    path = "input_data/pipes_diameters.txt"
    diameters = np.loadtxt(open(path, "rb"), delimiter = ",", usecols=(0))
    
    for p in pipes:
        d_opt = d[p].X / 1000
        # minimal diameter due to limitation of pipe friction
        d_min = dict_pipes["d_min"][p]
        # find index of next bigger norm-diameter in diameter list
        for index in range(len(diameters)):
            if diameters[index] >= d_min:
                index_min = index
                break
        # if optimal pipe diameter is smaller than lowest possible norm-diameter: choose lowest possible norm-diameter 
        if d_opt < diameters[index_min]:
            d_norm = diameters[index_min]
        # else: choose nearest available norm-diameter
        else:
            d_0 = diameters[index_min]
            d_1 = diameters[index_min+1]
            if abs(d_opt - d_0) < abs(d_opt - d_1):
                d_norm = d_0
            else:
                d_norm = d_1
        pipes_norm[p] = {"diameter": d_norm,                 # m,   norm diameter
                        "length": edge_lengths[p]
                        }
    
    
    # Print pipe diameters
    print("Pipe diameters:")
    for p in pipes:
        print("Pipe " + str(edge_dict_rev[p][0]) + "-" + str(edge_dict_rev[p][1]) + ": " + str(round(pipes_norm[p]["diameter"]*1000,2)) + " mm.")
        
   
    
    #%% RECALCULATE NETWORK TAC USING NORM PIPE DIAMETERS
    
    
    # Recalculate pipe costs with norm diameters
    inv_pipes_norm = sum((param["inv_earth_work"] + 2 * param["inv_pipe_PE"] * (pipes_norm[p]["diameter"]*1000)**2) * edge_lengths[p] for p in pipes)
    tac_pipes_norm = inv_pipes_norm * (param["ann_factor_pipe"] + param["cost_om_pipe"])
    
    # Calculate exact pump costs with norm diameters
    pump_el_norm = {}
    for p in pipes:
        d_norm = pipes_norm[p]["diameter"]
        pump_el_norm[p] = {}
        for line in ["plus", "minus"]:
            pump_el_norm[p][line] = np.zeros(8760)
            for t in time_steps:
                m_dot = dict_pipes["m_dot"][p][t]
                if m_dot >= 0:
                    pump_el_norm[p]["plus"][t] = prefac * 2 * edge_lengths[p] * m_dot**3/d_norm**5
                else:
                    pump_el_norm[p]["minus"][t] =  - prefac * 2 * edge_lengths[p] * m_dot**3/d_norm**5
    
    pump_caps_norm = sum(sum( np.max(pump_el_norm[p][line]) for line in ["plus", "minus"]) for p in pipes)
    pump_energy_total_norm = sum(sum(sum(pump_el_norm[p][line][t] for t in time_steps) for line in ["plus", "minus"]) for p in pipes) / 1000
    
    inv_pumps_norm = pump_caps_norm * param["inv_pump"]
    tac_pumps_norm = inv_pumps_norm * (param["ann_factor_pump"] + param["cost_om_pump"])
    
    
    
    # Recalculate annualized network costs
    tac_network_norm = (tac_pipes_norm + tac_pumps_norm + pump_energy_total_norm * param["price_el_pumps"]) / 1000
    
    
    # Store annualized network costs 
    param["tac_network"] = tac_network_norm
    
    
    # Calculate thermal losses and write them into parameters
    losses = soil.calculate_thermal_losses(param, pipes_norm)
    param["heat_losses"] = losses["heat"]
    param["cool_losses"] = losses["cool"]


    return param
        
 
    







