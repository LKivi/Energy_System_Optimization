# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:48:51 2018

@author: lkivi
"""

import numpy as np
import json
import pylab as plt
from pyproj import Proj, transform
import networkx as nx


#%%
# Design grid properties for the given parameters
def design_grid(param):
 
    # available standard pipe inner diameters (heating pipes: ISO 4200 / Set Pipes GmbH; cooling pipes: SDR11, PN16)
    param["diameters"] = {}
    path_heating = "input_data/pipes_heating.txt"
    path_cooling = "input_data/pipes_cooling.txt"
    param["diameters"]["heating"] = np.loadtxt(open(path_heating, "rb"), delimiter = ",", usecols=(0))                                                                               # m,   inner pipe diameters for cooling network
    param["diameters"]["cooling"] = np.loadtxt(open(path_cooling, "rb"), delimiter = ",", usecols=(0)) - 2 * np.loadtxt(open(path_cooling, "rb"), delimiter = ",", usecols=(1))          # m,   inner pipe diameters for cooling network
    
    data, graph = generateJson()
    dem = load_demands(data)
   
    # time series of heating supply temperatures according to heating curve
    param["T_heating_supply"] = get_T_supply(param)
    
    grid_types = ["heating", "cooling"]
    
    for typ in grid_types:
        
        for edge in data["edges"]:
            # get list of buildings supplied by that edge
            supplied_buildings = list_supplied_buildings(data, int(edge), graph)

            # sum up the demands of the buildings supplied by that edge        
            dem_buildings = np.zeros(8760)
            for building in supplied_buildings:
                 dem_buildings = dem_buildings + dem[typ][building]
                 
            
            # calculate time series of mass flowrates in the pipe
            m_flow = dem_buildings*1e6/(param["c_f"]*(abs(param["T_"+typ+"_supply"] - param["T_"+typ+"_return"])))
            
            # maximum mass flowrate
            m_flow_max = np.max(m_flow)
            data["edges"][edge]["max_flow_"+typ] = m_flow_max
            
            # calculate pipe diameter for given maxiumum pressure gradient
            d = ((8*m_flow_max**2*param["f_fric"])/(param["rho_f"]*np.pi**2*param["dp_pipe"]))**0.2
            
            # choose next bigger diameter from standard diameter list
            for d_norm in param["diameters"][typ]:
                if d_norm >= d:
                    d = d_norm
                    break
     
            # write pipe diameter into json array
            data["edges"][edge]["diameter_"+typ] = d
            
        
    # save new json-file in project folder
    with open("network.json", "w") as f: json.dump(data, f, indent=4, sort_keys=True)
    
    
    return data, param


#%%
# generate json-file of the network using the input files nodes.txt and edges.txt
# nodes.txt has to contain node latitudes, longitues, types (node, building or supply) and names
# edges.txt hat to contain names of start and end node for every edge
# pipe diameters are initialized with 0    
def generateJson():
    
    data_dict = {}
    
    path_nodes = "input_data/nodes.txt"     # contains node properties: latidude, longitude, name and type (supply, building, node)
    path_edges = "input_data/edges.txt"     #   
    
    nodes = {}
       
    lats = np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(0))                     # °,      node latitudes
    longs = np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(1))                    # °,      node longitudes
    types = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(2))   # --,       node type (node, building or supply)
    names = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(3))   # --,       node name
    
    for n in range(len(lats)):
        nodes[n] = {"lat": lats[n],
                    "lon": longs[n],
                    "type": types[n],
                    "name": names[n]
                    }
    
#    # Earth radius
#    r = 6371000
#    
#    # supply node serves as reference node (x=0, y=0)
##    for i in np.arange(np.size(nodes["lat"])):
##        if nodes["type"][i] == "supply":
##            lat_ref = nodes["lat"][i]
##            long_ref = nodes["long"][i]
#    
#    # find minimal lat/long
#    lat_ref = np.min(nodes["lat"])
#    long_ref = np.min(nodes["long"])
#    
#    # transform lat/long to xy-coordinates 
#    nodes["x"] = r*np.arccos(np.sin(nodes["lat"])**2 + np.cos(nodes["lat"])**2 * np.cos(nodes["long"] - long_ref))
#    nodes["y"] = r*np.arccos(np.sin(nodes["lat"])*np.sin(lat_ref) + np.cos(nodes["lat"])*np.cos(lat_ref))
#    # replace nan entries by 0
#    nodes["x"] = np.nan_to_num(nodes["x"])
#    nodes["y"] = np.nan_to_num(nodes["y"])

#    # shift x/y-coordinates so that supply node is at x = 0, y = 0
#    for i in np.arange(np.size(nodes["x"])):
#        if nodes["type"][i] == "supply":
#            supply_x = nodes["x"][i]
#            supply_y = nodes["y"][i] 
#    nodes["x"] = nodes["x"] - supply_x
#    nodes["y"] = nodes["y"] - supply_y    
    
    
    # Convert to x,y-coordinates
    nodes = transform_coordinates(nodes)
    

    edges = {}
    nodes_from = np.genfromtxt(open(path_edges, "rb"),dtype = 'str', delimiter = ",", usecols=(0))
    nodes_to = np.genfromtxt(open(path_edges, "rb"),dtype = 'str', delimiter = ",", usecols=(1))
      

    for k in range(len(nodes_from)):
        
        # find node indices in node dictionary
        for n1 in range(len(nodes)):
            if nodes[n1]["name"] == nodes_from[k]:
                id_from = n1
                break
        for n2 in range(len(nodes)):
            if nodes[n2]["name"] == nodes_to[k]:
                id_to = n2
                break
    
        if id_from > id_to:
            node_ids = (id_to, id_from)
        else:
            node_ids = (id_from, id_to)
        
        length = ((nodes[id_from]["x"] - nodes[id_to]["x"])**2 + (nodes[id_from]["y"] - nodes[id_to]["y"])**2)**0.5
        edges[k] = {"node_names": nodes[id_from]["name"] + "-" + nodes[id_to]["name"],
                           "node_ids": node_ids,
                           "length": length,
                           "diameter_heating": 0,
                           "diameter_cooling": 0
                           }
    
    data_dict = {"nodes": nodes,
                 "edges": edges}
    
        
    # save json-file in project folder
    with open("network.json", "w") as f: json.dump(data_dict, f, indent=4, sort_keys=True)
    
    graph = get_graph(data_dict)
    plot_grid(graph)
    
    return data_dict, graph
    
 
#%%  Draw network and save plot  
def plot_grid(graph):
 
    data = json.loads(open('network.json').read())
    
    pos = nx.get_node_attributes(graph, "pos")
#    nx.draw(graph, pos, with_labels=False, font_weight="bold")
    
    # Draw standard nodes black and small
    small_nodes = []
    for n in data["nodes"]:
        if data["nodes"][n]["type"] == "node":
            small_nodes.append(int(n))
    nx.draw_networkx_nodes(graph,pos,nodelist=small_nodes, node_size = 20, node_color="black")
    
    # Draw buildings blue
    buildings = []
    for n in data["nodes"]:
        if data["nodes"][n]["type"] == "building":
            buildings.append(int(n))
    nx.draw_networkx_nodes(graph,pos,nodelist=buildings, node_size = 100, node_color="blue")
    
    # Draw supply red
    supply = []
    for n in data["nodes"]:
        if data["nodes"][n]["type"] == "supply":
            supply.append(int(n))
    nx.draw_networkx_nodes(graph,pos,nodelist=supply, node_size = 120, node_color="red")
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos)
    
    
    plt.grid(True)
    plt.axis("equal")
    
    plt.show()
    plt.savefig("network.png")



#%%
# finds x- and y-coordinate of a node out of json file by name
#def findXY(data, name):
#    
#    found = 0
#    
#    for item in data["nodes"]:
#        
#        if item["name"] == name:
#            x = item["x"]
#            y = item["y"]
#            found = 1
#            
#    if found == 0:
#        print("Can't retrieve node coordinates to plot grid edges")
#        exit()
#        
#    return x,y


#%% create networkx-graph out of json-file
def get_graph(data):
    
    # get networkx-graph
    graph = nx.Graph()
    for k in data["nodes"]:
        graph.add_node(int(k), pos=(data["nodes"][k]["x"], data["nodes"][k]["y"]))
    
    ebunch = []
    for k in data["edges"]:
        ebunch.append((data["edges"][k]["node_ids"][0], data["edges"][k]["node_ids"][1]))   
    graph.add_edges_from(ebunch)
    
    return graph



#%% finds all buildings that are supplied by the considered edge
def list_supplied_buildings(data, edge_id, graph):

    supplied_buildings = []
    
    # get building ids and names
    dict_buildings = {}
    for k in data["nodes"]:
        if data["nodes"][k]["type"] == "building":
            dict_buildings[int(k)] = data["nodes"][k]["name"]
    
    # get supply node id
    for k in data["nodes"]:
        if data["nodes"][k]["type"] == "supply":
            supply_node = int(k)
            break   
    
    # get names of buildings which are supplied by this edge
    for building in dict_buildings:
        path = nx.shortest_path(graph,source = supply_node, target = building)
        # get edges along the path from supply to the building
        for k in range(len(path)-1):
            if path[k] < path[k+1]:
                edge_path = (path[k], path[k+1])
            else:
                edge_path = (path[k+1], path[k])
            # check if the considered edge is lies on that path
            if edge_path == data["edges"][edge_id]["node_ids"]:
                supplied_buildings.append(dict_buildings[building])
    
#    print(edge_id)
#    print(supplied_buildings)
    
    return supplied_buildings
            
    
    
#    # initialize array of end points with end point of the input edge
#    endings = [edge["node_1"]]
#    
#    # initialize list of buildings
#    supplied_buildings = []
#    
#    for i in range(1000):
#        
#        # check if the found ending points are buildings; add the found buildings to the buildings-array
#        for iEnding in range(np.size(endings)):
#            nodeName = endings[iEnding]
#            for item in data["nodes"]:
#                if item["name"] == nodeName and item["type"] == "building":
#                    supplied_buildings.append(nodeName)        
#        
#        # set end points to new start points
#        starts = endings
#         
#        #reset ending nodes
#        endings = []
#        
#        #find all edges beginning with any entry of starts and get their ending points
#        for iStart in range(np.size(starts)):
#            nodeName = starts[iStart]
#        
#            for item in data["edges"]:
#                if item["node_0"] == nodeName:
#                    endings.append(item["node_1"])
#        
#        # if no new edges are found, the buildings array is returned
#        if endings == []:
#            return supplied_buildings
 

#%% loads demand arrays
def load_demands(data):
    
    path_demands = "input_data/demands/"
    dem = {}
    dem["heating"] = {}
    dem["cooling"] = {}
    
    dem["heating"]["sum"] = np.zeros(8760)
    dem["cooling"]["sum"] = np.zeros(8760)
    
    # collect building names out of json-data
    buildings = []
    for k in data["nodes"]:
        if data["nodes"][k]["type"] == "building":
            buildings.append(data["nodes"][k]["name"])
    
    # get loads of each building and sum up 
    for name in buildings:
        dem["heating"][name] = np.loadtxt(open(path_demands + name + "_heating.txt", "rb"), delimiter = ",", usecols=(0))/1000      # MW,   heating load of building
        dem["heating"]["sum"] = dem["heating"]["sum"] + dem["heating"][name]
        dem["cooling"][name] = np.loadtxt(open(path_demands + name + "_cooling.txt", "rb"), delimiter = ",", usecols=(0))/1000      # MW,   cooling load of building
        dem["cooling"]["sum"] = dem["cooling"]["sum"] + dem["cooling"][name]
    
    
    return dem
        

#%%
def get_T_supply(param):
 
    path_weather = "input_data/weather.csv"
    T_amb = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(0))
    
    
    if not param["switch_low_temp"]:
        
        T_supply = np.zeros(8760)
    
        for i in range(np.size(T_amb)):   
            if T_amb[i] < -15:
                T_supply[i] = 140
            elif T_amb[i] < -10:
                T_supply[i] = 140 - 17/5*(T_amb[i]+15)
            elif T_amb[i] < 2:
                T_supply[i] = 123
            elif T_amb[i] < 15:
                T_supply[i] = 123 - 28/13*(T_amb[i]-2)
            else:
                T_supply[i] = 95
    
    else:
        
        T_supply = np.ones(8760) * param["T_heating_supply_low"]
    
#    plt.plot(range(8760), T_supply)
#    plt.show()
    
    return T_supply

  
#%%
def transform_coordinates(nodes):
    outProj = Proj(init='epsg:25832')   # ETRS89 / UTM zone 32N
    inProj = Proj(init='epsg:4258')     # Geographic coordinate system: EPSG 4326
    
    # get x- and y- coordinates and find minimal values of each
    min_x, min_y = transform(inProj,outProj,nodes[0]["lon"],nodes[0]["lat"])
    for n in range(len(nodes)):
        nodes[n]["x"],nodes[n]["y"] = transform(inProj,outProj,nodes[n]["lon"],nodes[n]["lat"])
        if nodes[n]["x"] < min_x:
            min_x = nodes[n]["x"]
        if nodes[n]["y"] < min_y:
            min_y = nodes[n]["y"]
    
    # Shift coordinate system by minimal x- and y- value        
    for n in range(len(nodes)):
        nodes[n]["x"] = nodes[n]["x"] - min_x
        nodes[n]["y"] = nodes[n]["y"] - min_y
        
    return nodes


#%%
def design_pump(data, param, dem_buildings):
    
    graph = get_graph(data)
    
    # get supply unit
    for n in data["nodes"]:
        if data["nodes"][n]["type"] == "supply":
            supply_node = int(n) 
    
    # get path lengths to buildings
    dict_lengths = {}
    for n in data["nodes"]:
        if data["nodes"][n]["type"] == "building":
           path = nx.shortest_path(graph, source = supply_node, target = int(n))
           path_length = 0
           for k in range(len(path)-1):
               path_length += np.sqrt((data["nodes"][path[k+1]]["x"]-data["nodes"][path[k]]["x"])**2 + (data["nodes"][path[k+1]]["y"]-data["nodes"][path[k]]["y"])**2)
           dict_lengths[data["nodes"][n]["name"]] = path_length
           
    # Calculate pressure loss on every path at maximum load and find the path with maximum pressure loss and calculate pump capacities
    pump_caps = {}
    for typ in ["heating", "cooling"]:
        pressure_losses = []
        for destination in dict_lengths:
            max_dem = np.max(dem_buildings[typ][destination])
            dp = (max_dem > 0) * ((2*param["dp_pipe"]*dict_lengths[destination])/(1 - param["dp_single"]) +        # pressure loss in pipes
                          max_dem * param["dp_substation"])                                                        # pressure drop in substation
            pressure_losses.append(dp)
        dp_max = max(pressure_losses)
      
        # find edge at supply unit to get mass flow through pump
        for e in data["edges"]:
            n1 = data["edges"][e]["node_ids"][0]
            n2 = data["edges"][e]["node_ids"][1]
            if data["nodes"][n1]["name"] == "supply" or data["nodes"][n2]["name"] == "supply":
                m_flow_pump = data["edges"][e]["max_flow_"+typ]
    
        cap = m_flow_pump/(param["eta_pump"]*param["rho_f"])*dp_max / 1e6             # MW,   electrical power of pump
        pump_caps[typ] = cap
        param["pump_cap_"+typ] = cap
        
    print(pump_caps)
        
        
    return param
      