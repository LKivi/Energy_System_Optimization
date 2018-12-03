# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:28:12 2018

@author: mwi
"""

import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def get_edge_str():
    return ["0-1", "0-2", "0-3", "1-2", "1-3", "2-3"]

def sum_intergers_to(n):
    return int(0.5 * (n*n + n))

# Create a new model
model = gp.Model("Ectogrid_topology")

# Implement clustering!!!

# Specific costs
c_fix = 0
c_var = 1
    
nodes = {}
nodes[0] = {}
nodes[0]["loc"] = (10, 15)
nodes[0]["dem"] = {0: -1, 1: 3}
nodes[1] = {}
nodes[1]["loc"] = (6, 21)
nodes[1]["dem"] = {0: -1, 1: 3}
nodes[2] = {}
nodes[2]["loc"] = (8, 22)
nodes[2]["dem"] = {0: 1, 1: -2}
nodes[3] = {}
nodes[3]["loc"] = (12, 17)
nodes[3]["dem"] = {0: -1, 1: 2}

time_steps = range(len(nodes[0]["dem"]))
    
#%% CREATE VARIABLES

edges = get_edge_str()#(sum_intergers_to(len(nodes)-1)) 
node_list = range(len(nodes))

number_of_feasible_balancing_units = 1

# Purchase decision binary variables (1 if device is installed, 0 otherwise)
x = model.addVars(edges, vtype="B", name="x")
    
ksi = model.addVars(node_list, vtype="B", name="balancing_unit")
    
# Edge capacity
cap = model.addVars(edges, vtype="C", name="nominal_edge_capacity")  # für jedes Element in edges
 
m_dot = model.addVars(edges, time_steps, vtype="C", lb=-100, name="mass_flow_pipe")

m_bal = model.addVars(node_list, time_steps, vtype="C", lb=-100, name="mass_flow_balancing")

d_powered = model.addVars(edges, vtype = "C", name = "d_powered")   # edge inner diameter to the power of 5/2

obj = model.addVar(vtype="C", name="total_costs")
    

# Define objective function
model.update()
model.setObjective(sum(x[edge] * c_fix + d_powered[edge] * c_var for edge in edges), gp.GRB.MINIMIZE)

#%% CONSTRAINTS
# Node balance
for t in time_steps:
    model.addConstr( m_dot["0-1",t] + m_dot["0-2",t] + m_dot["0-3",t] + nodes[0]["dem"][t] + m_bal[0,t] == 0)
    model.addConstr(-m_dot["0-1",t] + m_dot["1-2",t] + m_dot["1-3",t] + nodes[1]["dem"][t] + m_bal[1,t] == 0)
    model.addConstr(-m_dot["0-2",t] - m_dot["1-2",t] + m_dot["2-3",t] + nodes[2]["dem"][t] + m_bal[2,t] == 0)
    model.addConstr(-m_dot["0-3",t] - m_dot["1-3",t] - m_dot["2-3",t] + nodes[3]["dem"][t] + m_bal[3,t] == 0)
        
# Maximum number of balancing units
model.addConstr(sum(ksi[node] for node in node_list) <= number_of_feasible_balancing_units, "number_balancing_units")

# Balancing power is only at balancing nodes possible
for node in node_list:
    for t in time_steps:
        model.addConstr(m_bal[node,t] <= ksi[node] * 1000)
        model.addConstr(-m_bal[node,t] <= ksi[node] * 1000)
        
# Help constraint
model.addConstr(ksi[3] == 1)
    
# Mass flow on edge must not exceed pipe capacity   
for edge in edges:
    for t in time_steps:
        model.addConstr(m_dot[edge,t] <= cap[edge])
        model.addConstr(-m_dot[edge,t] <= cap[edge])
        model.addConstr(m_dot[edge,t] <= x[edge] * 1000)
        model.addConstr(-m_dot[edge,t] <= x[edge] * 1000)


# diamter constraint due to limitation of allowed pressure gradient
C = (8 * 0.025)/(1000 * np.pi**2 * 150)
for edge in edges:
     model.addConstr(d_powered[edge] >=  C * cap[edge])
     model.addConstr(d_powered[edge] <=  x[edge] * 1000)

#%% RUN OPTIMIZATION
        
model.optimize()


#%% EVALUATE RESULTS

if model.Status in (3,4) or model.SolCount == 0:  # "INFEASIBLE" or "INF_OR_UNBD"
        model.computeIIS()
        model.write("model.ilp")
        print('Optimization result: No feasible solution found.')
    
else:
    model.write("model.lp")
    model.write("model.prm")
    model.write("model.sol")
    
    print("Optimization done.\n")
    for node in node_list:
        if ksi[node].X == 1:            # Variablenattribut .X --> Wert der Variablen in der aktuellen Lösung
            balancing_node = node
            print("Balancing unit installed at node: " + str(node))
            for t in time_steps:
                print("Balancing power: t" + str(t) + ": " + str(round(m_bal[node,t].X,2)))
            
    print("Pipe capacities:")
    for edge in edges:
        print("Edge" + edge + ": " + str(round(cap[edge].X,2)) + " [x:" + str(x[edge].X) + "]")
        
    print("\nMass flows capacities:")    
    for t in time_steps:
        for edge in edges:
            print("m_dot" + edge + "_t" + str(t) + ": " + str(round(m_dot[edge,t].X,2)))
        print("\n")
                    
graph = nx.Graph()
for k in node_list:
    graph.add_node(k,pos=nodes[k]["loc"])

graph.add_weighted_edges_from([(0, 1, cap["0-1"].X), (0, 2, cap["0-2"].X), (0, 3, cap["0-3"].X), (1, 2, cap["1-2"].X),
                               (1, 3, cap["1-3"].X), (2, 3, cap["2-3"].X)])

# Plot pipe installation in black
pos = nx.get_node_attributes(graph, "pos")
weights = [3*graph[u][v]["weight"] for u,v in graph.edges()]
plt.subplot(121)
nx.draw(graph, pos, with_labels=True, font_weight="bold", width=weights)  # schwarze Linien = Gesamtkapazität


# Plot time steps in red
t = 0
graph = nx.Graph()
for k in node_list:
    graph.add_node(k,pos=nodes[k]["loc"])
graph.add_weighted_edges_from([(0, 1, m_dot["0-1",t].X), (0, 2, m_dot["0-2",t].X), (0, 3, m_dot["0-3",t].X), 
                               (1, 2, m_dot["1-2",t].X),  (1, 3, m_dot["1-3",t].X), (2, 3, m_dot["2-3",t].X)])

weights = [3*graph[u][v]["weight"] for u,v in graph.edges()]
nx.draw(graph, pos, font_weight="bold", width=weights, edge_color="red")   # rote Linie = Auslastung zum aktuellen Zeitschritt

# Highlight balancing unit
nx.draw_networkx_nodes(graph,pos,nodelist=[balancing_node],node_color="blue")
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.axis("equal")

plt.subplot(122)
graph = nx.Graph()
for k in node_list:
    graph.add_node(k,pos=nodes[k]["loc"])
    


graph.add_weighted_edges_from([(0, 1, cap["0-1"].X), (0, 2, cap["0-2"].X), (0, 3, cap["0-3"].X), (1, 2, cap["1-2"].X),
                               (1, 3, cap["1-3"].X), (2, 3, cap["2-3"].X)])
pos = nx.get_node_attributes(graph, "pos")
weights = [3*graph[u][v]["weight"] for u,v in graph.edges()]
nx.draw(graph, pos, with_labels=True, font_weight="bold", width=weights)

t = 1
graph = nx.Graph()
for k in node_list:
    graph.add_node(k,pos=nodes[k]["loc"])
graph.add_weighted_edges_from([(0, 1, m_dot["0-1",t].X), (0, 2, m_dot["0-2",t].X), (0, 3, m_dot["0-3",t].X), 
                               (1, 2, m_dot["1-2",t].X),  (1, 3, m_dot["1-3",t].X), (2, 3, m_dot["2-3",t].X)])

weights = [3*graph[u][v]["weight"] for u,v in graph.edges()]
nx.draw(graph, pos, font_weight="bold", width=weights, edge_color="red")
# Highlight balancing unit
nx.draw_networkx_nodes(graph,pos,nodelist=[balancing_node],node_color="blue")
#ax.grid(color='r', linestyle='-', linewidth=2)

plt.axis("equal")


nx.write_gml(graph, "graph.gml")
plt.show()
          
          