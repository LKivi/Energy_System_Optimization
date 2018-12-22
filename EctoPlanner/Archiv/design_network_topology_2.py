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


## ALTERNATIVE OPTIMIZATION: MINIMIZATION OF TOTAL PIPE LENGTH

def design_network(nodes, param, time_steps, dir_results):
    
#    print("HIER MUESSEN NOCH PUMPARBEITEN BERUECKSICHTIGT WERDEN in der ZielFn, sonst hat man keinen vernuenftigen Trade-Off zwischen Invest und Betrieb.")

    start_time = time.time()
    
    # Create a new model
    model = gp.Model("Ectogrid_topology")
    
        
    #%% CREATE VARIABLES
    
    edge_dict, edge_dict_rev, edges, compl_graph, edge_list, edge_lengths = par.get_edge_dict(len(nodes),nodes)
    param = par.calc_pipe_costs(nodes, edges, edge_dict_rev, param)
    node_list = range(len(nodes))
    
#    allowed_edges, banned_edges = par.ban_edges(param, nodes, compl_graph, edge_dict, edge_dict_rev, edges)
#    dict_clusters = par.cluster_nodes(nodes, param)
#    dict_connections = par.cluster_connections(dict_clusters, edge_dict)
    
    x = model.addVars(edges, vtype="B", name="x") # Purchase decision binary variables (1 if device is installed, 0 otherwise)
        
    ksi = model.addVars(node_list, vtype="B", name="balancing_unit") # Binary decision: balancing unit installed in node
    
    cap = model.addVars(edges, vtype="C", name="nominal_edge_capacity") # Mass flow capacity of edge
     
    m_dot = model.addVars(edges, time_steps, vtype="C", lb=-1000, name="mass_flow_pipe") # Mass flow in pipe in every time step (from 0->1 positive)
    
    m_bal = model.addVars(node_list, time_steps, vtype="C", lb=-1000, name="mass_flow_balancing") # Mass flow from warm supply pipe to cold return pipe
    
    total_length = model.addVar(vtype="C", name="total_pipe_length") # Objective function
        
    
    #%% DEFINE OBJECTIVE FUNCTION
    model.update()
    model.setObjective(total_length, gp.GRB.MINIMIZE)
    
    #%% CONSTRAINTS
    
    # objective
    model.addConstr(total_length == sum(x[edge] * edge_lengths[edge] for edge in edges))
    
    # ban edges
#    for edge in banned_edges:
#        model.addConstr(x[edge] == 0)
    
#    for edge in edges:
#        model.addConstr(x[edge] == 1)
    
    # only one connection between each pair of clusters is allowed
#    for item in dict_connections:
#        model.addConstr(sum(x[dict_connections[item][k]] for k in range(len(dict_connections[item]))) <= 1)
    
    # Node balance   
    for node in node_list:
        edges_minus = []
        edges_plus = []
        
        nodes_lower = np.arange(node)                       # nodes which have lower ID-number
        nodes_higher = np.arange(node+1, len(node_list))    # nodes which have higher ID-number
        
        for node_from in nodes_lower:
            edges_plus.append(edge_dict[(node_from, node)])
        for node_to in nodes_higher:
            edges_minus.append(edge_dict[(node, node_to)])
            
        for t in time_steps:
            model.addConstr( sum(m_dot[k,t] for k in edges_plus) - sum(m_dot[k,t] for k in edges_minus) + m_bal[node,t] == nodes[node]["mass_flow"][t] )
    
#    for node in node_list:
#        for t in time_steps:
#            model.addConstr( -m_dot[0,t] + m_bal[0,t] == nodes[0]["mass_flow"][t])
#            model.addConstr(m_dot[0,t] + m_bal[1,t] == nodes[1]["mass_flow"][t])

    
#    list_edge_id_used = []
#    for node in node_list:
#        adjacent_edges = list(compl_graph.edges(node, data=False))
#        ids_plus_sign = []
#        ids_minus_sign = []
#        for e in adjacent_edges:
#            if e[0] > e[1]:
#                e = (e[1], e[0])
#            edge_id = edge_dict[e]
#            if edge_id not in list_edge_id_used:
#                ids_plus_sign.append(edge_id)
#                list_edge_id_used.append(edge_id)
#            else:
#                ids_minus_sign.append(edge_id)
#            
#        for t in time_steps:
#            model.addConstr(sum(m_dot[k,t] for k in ids_plus_sign) - sum(m_dot[k,t] for k in ids_minus_sign) 
#                            + nodes[node]["mass_flow"][t] - m_bal[node,t] == 0)                                         # mass_flows kommen aus Intra-Balancing der Gebäude, wird hier als bekannt vorausgesetzt
            
    # Maximum number of balancing units
    model.addConstr(sum(ksi[node] for node in node_list) <= param["number_of_balancing_units"], "number_balancing_units")
    
    # Balancing power is only at balancing nodes possible
    for node in node_list:
        for t in time_steps:
            model.addConstr(m_bal[node,t] <= ksi[node] * 1000)
            model.addConstr(-m_bal[node,t] <= ksi[node] * 1000)
            
    # Help constraint
#    balancing_node = 0
#    model.addConstr(ksi[balancing_node] == 1)

        
    # Mass flow on edge must not exceed pipe capacity   
    for edge in edges:
        for t in time_steps:
            model.addConstr(m_dot[edge,t] <= cap[edge])
            model.addConstr(-m_dot[edge,t] <= cap[edge])
            model.addConstr(cap[edge] <= x[edge] * 1000)
            
    
    # diamter constraint due to limitation of allowed pressure gradient
#    C = ((8 * param["f_fric"])/(param["rho_f"] * np.pi**2 * param["dp_pipe"])) ** 0.5 * (100 ** 2.5)
#    for edge in edges:
#        model.addConstr(d_powered[edge] >=  C * cap[edge])
#        model.addConstr(d_powered[edge] <=  x[edge] * 1000)
    
    
    #%% RUN OPTIMIZATION
    print("Precalculation and model set up done in %f seconds." %(time.time() - start_time))
    
    # Set solver parameters
    model.Params.MIPGap       = param["MIPGap"]             # ---,  gap for branch-and-bound algorithm
    model.Params.method       = 2                           # ---, -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc. (only affects root node)
    model.Params.Heuristics   = 0                           # Percentage of time spent on heuristics (0 to 1)
    model.Params.MIPFocus     = 2                           # Can improve calculation time (values: 0 to 3)
    model.Params.Cuts         = 2                           # Cut aggressiveness (values: -1 to 3)
    model.Params.PrePasses    = 8                           # Number of passes performed by presolving (changing can improve presolving time) values: -1 to inf
#    model.Params.NumericFocus = 2
    
    # Execute calculation
    start_time = time.time()        
   
    
    model.optimize()
    
    
    #%% EVALUATE RESULTS

    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    if model.Status in (3,4) or model.SolCount == 0:  # "INFEASIBLE" or "INF_OR_UNBD"
#            model.computeIIS()
#            model.write("model.ilp")
            print('Optimization result: No feasible solution found.')
        
    else:
        
        # save results
        model.write(dir_results + "\\model.lp")
        model.write(dir_results + "\\model.prm")
        model.write(dir_results + "\\model.sol")
        
        print("Optimization done.\n")
        model.printQuality()
        
        # write node data into json-files
        # Convert numpy arrays to normal lists
        for n in nodes:
            for key in nodes[n]:
                if type(nodes[n][key]) == np.ndarray:
                   nodes[n][key] = nodes[n][key].tolist()
        # create dict with node names and positions only
        node_positions = {}
        for n in nodes:
            node_positions[n] = {}
            for key in ["name", "x", "y"]:
                node_positions[n][key] = nodes[n][key]            
        # save json files
        nodes = {**nodes}
        node_positions = {**node_positions}
        with open(dir_results + "\\nodes.json", "w") as outfile:
            json.dump(nodes, outfile, indent=4, sort_keys=True)
        with open(dir_results + "\\node_positions.json", "w") as outfile:
            json.dump(node_positions, outfile, indent=4, sort_keys=True)
        
        
        for node in node_list:
            print(ksi[node].X)
            if ksi[node].X == 1:            # Variablenattribut .X --> Wert der Variablen in der aktuellen Lösung
                balancing_node = node
                print("Balancing unit installed at node: " + str(node))
                print("Max BU heating: " + str(max(m_bal[node,t].X for t in time_steps)) + " kg/s")
                print("Min BU heating: " + str(min(m_bal[node,t].X for t in time_steps)) + " kg/s")
                
#                for t in time_steps:
#                    print("Balancing power: t" + str(t) + ": " + str(round(m_bal[node,t].X,2)))
                
        print("Pipe lengths:")
        for edge in edges:
            print("Edge " + str(edge_dict_rev[edge][0]) + "-" + str(edge_dict_rev[edge][1]) + ": "  + str(round(edge_lengths[edge], 2)) + " m [x:" + str(abs(np.round(x[edge].X))) + "]")
            
#        print("\nMass flows capacities:")    
#        for t in time_steps:
#            for edge in edges:
#                print("m_dot" + edge + "_t" + str(t) + ": " + str(round(m_dot[edge,t].X,2)))
#            print("\n")
            
            
        graph = nx.Graph()
        for k in node_list:
            graph.add_node(k, pos=(nodes[k]["x"], nodes[k]["y"]))
        
        ebunch = []
        for k in range(len(edge_dict_rev)):
            if round(x[k].X,0) == 1:
                ebunch.append((edge_dict_rev[k][0], edge_dict_rev[k][1]))
        
        graph.add_edges_from(ebunch)
        
        # Plot pipe installation in black
        pos = nx.get_node_attributes(graph, "pos")
        nx.draw(graph, pos, with_labels=True, font_weight="bold")
        
        # Highlight balancing unit
        nx.draw_networkx_nodes(graph,pos,nodelist=[balancing_node],node_color="blue")
        
        plt.axis([40, 160, 0, 0.03])
        plt.grid(True)
        plt.axis("equal")
        
        plt.savefig(dir_results + "\\graph.png")
        
#        plt.show()
        plt.clf()
        
    #%% Create plots and result files
#    print("Use tuples as index of edges")
#    post_processing.save_network_data(nodes, cap)
#    post_processing.plot_network(nodes, cap)
    