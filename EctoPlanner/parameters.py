# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:12:14 2018

@author: mwi
"""


from pyproj import Proj, transform
import math
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import random
import clustering_medoid as clustering


def load_params(use_case, path_file, scenario):
    
    assert (use_case != "FZJ" or use_case != "EON" or use_case != "simple_test"), "Use case '" + use_case + "' not known."
    path_input = path_file + "\\input_data\\" + use_case + "\\"
    print("Using data set: '" + use_case + "'")
    
    time_steps = range(8760)
     
        
    if use_case == "FZJ":
        
        
        # load node data 
        path_nodes = path_input + "nodes.txt"
        path_demands = path_input + "demands\\"
        latitudes =  np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(0))                       # °,        node latitudes
        longitudes =  np.loadtxt(open(path_nodes, "rb"), delimiter = ",", usecols=(1))                      # °,        node latitudes
        names = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(3))           # --,       node names
                
        # Fill node-dict
        nodes = {}
        for index in range(len(latitudes)):
            nodes[index] = {
                            "number": index,
                            "lat": latitudes[index],
                            "lon": longitudes[index],
                            "name": names[index],
                            "heat": np.loadtxt(open(path_demands + names[index] + "_heating.txt", "rb"),delimiter = ",", usecols=(0)),                    # kW, heat demand
                            "cool": np.loadtxt(open(path_demands + names[index] + "_cooling.txt", "rb"),delimiter = ",", usecols=(0)),                    # kW, cooling demand                                                                                    # °C, heating return temperature
                            }
        
         
        # Check small demand values
        for n in nodes:
            for t in range(8760):
                if nodes[n]["heat"][t] < 0.01:
                    nodes[n]["heat"][t] = 0
                if nodes[n]["cool"][t] < 0.01:
                    nodes[n]["cool"][t] = 0
                    

        # Heating temperatures
        for n in nodes:
            nodes[n]["T_heating_supply"] = 60 * np.ones(8760)
            nodes[n]["T_heating_return"] = 30 * np.ones(8760)       
        
        # Cooling temperatures
        for n in nodes:
             # Data centres have higher cooling return temperatures 
             if nodes[n]["name"] in ["16.3", "16.4"]:
                 nodes[n]["T_cooling_supply"] = 16 * np.ones(8760)
                 nodes[n]["T_cooling_return"] = 30 * np.ones(8760)
             else:
                 nodes[n]["T_cooling_supply"] = 16 * np.ones(8760)
                 nodes[n]["T_cooling_return"] = 20 * np.ones(8760)                   
             
        # Maximum roof areas for PV installation and maximum thermal storage sizes
        for n in nodes:
             # Data centres have higher cooling return temperatures 
             if nodes[n]["name"] in ["16.3", "16.4", "04.1"]:
                 nodes[n]["area_PV_max"] = 800     # m^2
                 nodes[n]["V_TES_max"] = 25         # m^3                 
             else:
                 nodes[n]["area_PV_max"] = 400  # m^2
                 nodes[n]["V_TES_max"] = 10     # m^3  
                 
            
            
        # Transform coordinates to x,y        
        nodes = transform_coordinates(nodes)
        


#%% GENERAL PARAMETERS
    param = {
            "interest_rate":  0.05,        # ---,          interest rate
             "observation_time": 20.0,      # a,            project lifetime
               
             # Gap for branch-and-bound
             "MIPGap":      1e-5,           # ---,          MIP gap            
             
             
             # Parameter switches
             "switch_variable_price": 1,            # ---,      1: variable electricity price
             "switch_var_revenue": 1,               # ---,      1: variable feed-in revenue                                                                              
             "switch_COP_buildings": 1,             # ---,      1: Use COP model by Jensen et al. for building devices ; 0: Use COP correlation derived from NIBE F1345 for building devices
             "switch_cost_functions": 0,            # ---,      1: Use piece-wise linear cost functions, 0: Use constant investment costs (kEUR/MW) for BU devices
             
             # Number of type-days
             "n_clusters": 50,                      
             

             # Building devices             
             "use_eh_in_bldgs": 1,                 # ---,          should electric heaters be used in buildings?
             "use_boi_in_bldgs": 1,                # ---,          should boilers be used in buildings?
             "use_frc_in_bldgs": 1,                # ---,          should free coolers be used in buildings?
             "use_air_in_bldgs": 1,                # ---,          should air coolers be used in buildings?
             "use_cc_in_bldgs": 1,                 # ---,          should compression chillers be used in buildings?
             "use_pv_in_bldgs": 1,                 # ---,          should pv cells be used in buildings?
             "use_hp_in_bldgs": 1,
#             "use_tes_in_bldgs": 1,
             
#             "op_hours_el_heater": 500,    # h,            hours in which the eletric heater is operated
             
             # BU Devices
             "feasible_TES": 1,             # ---,          are thermal energy storages feasible for BU?
             "feasible_BAT": 0,             # ---,          are batteries feasible for BU?
             "feasible_CTES": 1,            # ---,           are cold thermal energy storages feasible for BU?
             "feasible_BOI": 1,             # ---,          are gas-fired boilers feasible for BU?
             "feasible_CHP": 1,             # ---,          are CHP units feasible for BU?
             "feasible_EH": 0,              # ---,          are electric heater feasible for BU?
             "feasible_CC": 1,              # ---,          are compression chiller feasible for BU?
             "feasible_AC": 1,              # ---,          are absorption chiller feasible for BU?
             "feasible_HYB": 0,             # ---,          are hybrid coolers feasible for BU?
             "feasible_HP": 1               # ---,          are heat pumps feasible for BU?
             }
    
    
    # Assign scenario switches according to scenario name
    param_switches = {}
    # Type-day clustering
    if "typedays" in scenario:
        param_switches["switch_clustering"] = 1
    else:
        param_switches["switch_clustering"] = 0
    # Thermal balances for BU design
    if "absC" in scenario:
        param_switches["switch_single_balance"] = 0
    else:
        param_switches["switch_single_balance"] = 1
    # Thermal building storages
    if "noBldgTES" in scenario:
        param_switches["use_tes_in_bldgs"] = 0
    else:
        param_switches["use_tes_in_bldgs"] = 1
    # Sand-alone scenario
    if "standalone" in scenario:
        param_switches["switch_stand_alone"] = 1
    else:
        param_switches["switch_stand_alone"] = 0
    
    
    param.update(param_switches)


    #%% WEATHER DATA
    
    param["t_air"] = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(0))        # °C,    Air temperatur 
    param["G_sol"] = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(3))        # W/m^2  Solar radiation  
    

    #%% PIPE TEMPERATURES
    param_temperatures = {"T_hot": 18 * np.ones(8760),      # °C,   hot pipe temperature
                          "T_cold": 14 * np.ones(8760),     # °C,   cold pipe temperature
                          }    
    
    param.update(param_temperatures)
    
    
    #%% DEEP GROUND TEMPERATURE FOR GEOTHERMAL USE
    
    param["T_soil_deep"] = 10 * np.ones(8760)               # °C
    
    
    
    #%% ELECTRICITY PRICES
    
    # Price for electricity taken from grid
    if param["switch_variable_price"]:
        spot_prices = np.loadtxt(open("input_data/Spotpreise15.txt", "rb"), delimiter = ",",skiprows = 0, usecols=(0)) / 1000        # kEUR/MWh 
        param["price_el"] = 0.10808 + spot_prices       # kEUR/MWh
    else:
        param["price_el"] = 0.14506 * np.ones(8760)     # kEUR/MWh
    
    
    # Feed-in revenue
    param["revenue_feed_in"] = {}
    if param["switch_var_revenue"]:
        param["revenue_feed_in"]["CHP"] = np.loadtxt(open("input_data/revenue_feed_in.txt", "rb"), delimiter = ",",skiprows = 0, usecols=(0)) / 1000        # kEUR/MWh 
        param["revenue_feed_in"]["PV"] = np.loadtxt(open("input_data/revenue_feed_in.txt", "rb"), delimiter = ",",skiprows = 0, usecols=(1)) / 1000         # kEUR/MWh  
    else:
        param["revenue_feed_in"]["CHP"] = 0.06 * np.ones(8760)         # kEUR/MWh                  # former: 0.06442
        param["revenue_feed_in"]["PV"] = 0.085 * np.ones(8760)         # kEUR/MWh 
        
    # Grid capacity price
    param["price_cap_el"] =  59.660                         # kEUR/MW        
    

 
    
    #%% GAS PRICES
    
    # Gas price per MWh
    param["price_gas"] = 0.02824     # kEUR/MWh
    
    # Grid capacity price
    param["price_cap_gas"] = 12.149  # kEUR/MW



    
    #%% CO2 EMISSIONS
    
    param["gas_CO2_emission"] = 0.2         # t_CO2/MWh,    specific CO2 emissions (natural gas)
    param["grid_CO2_emission"] = 0.503      # t_CO2/MWh,    specific CO2 emissions (grid)
    
    
    
    
    #%% CLUSTER TIME SERIES INTO TYPE-DAYS
    
    if param["switch_clustering"]:
    
        time_series = []
#        weight = []
        # Cluster time series into typical days
        for n in nodes:
            time_series.append(nodes[n]["heat"])
#            weight.append(sum(nodes[n]["heat"][t] for t in range(8760)))
            time_series.append(nodes[n]["cool"])
#            weight.append(sum(nodes[n]["cool"][t] for t in range(8760)))
            time_series.append(nodes[n]["T_heating_supply"])
            time_series.append(nodes[n]["T_heating_return"])
            time_series.append(nodes[n]["T_cooling_supply"])
            time_series.append(nodes[n]["T_cooling_return"])
        time_series.append(param["t_air"])
        time_series.append(param["G_sol"])
        time_series.append(param["T_hot"])
        time_series.append(param["T_cold"])
        time_series.append(param["T_soil_deep"])
        time_series.append(param["price_el"])
        time_series.append(param["revenue_feed_in"]["CHP"])
        time_series.append(param["revenue_feed_in"]["PV"])
        
        
        inputs_clustering = np.array(time_series)
                                                                   
        (clustered_series, nc, z) = clustering.cluster(inputs_clustering, 
                                         param["n_clusters"],
                                         norm = 2,
                                         mip_gap = 0.01, 
                                       #  weights=weight
                                         )
        
        # save frequency of typical days
        param["day_weights"] = nc
        param["day_matrix"] = z
        
        # Read clustered time series
        n_nodes = len(nodes)
        for n in nodes:
            nodes[n]["heat"] = clustered_series[6*n]
            nodes[n]["cool"] = clustered_series[6*n+1]
            nodes[n]["T_heating_supply"] = clustered_series[6*n+2]
            nodes[n]["T_heating_return"] = clustered_series[6*n+3]
            nodes[n]["T_cooling_supply"] = clustered_series[6*n+4]
            nodes[n]["T_cooling_return"] = clustered_series[6*n+5]
        param["t_air"] = clustered_series[6*n_nodes]
        param["G_sol"] = clustered_series[6*n_nodes+1]
        param["T_hot"] = clustered_series[6*n_nodes+2]
        param["T_cold"] = clustered_series[6*n_nodes+3]
        param["T_soil_deep"] = clustered_series[6*n_nodes+4]
        param["price_el"] = clustered_series[6*n_nodes+5]
        param["revenue_feed_in"]["CHP"] = clustered_series[6*n_nodes+6]
        param["revenue_feed_in"]["PV"] = clustered_series[6*n_nodes+7]
        


    
    #%% SOIL PARAMETERS   
    param_soil = {"alpha_soil": 0.8,                           #---,       soil surface absorptance
                  "epsilon_soil": 0.9,                         #---,       soil surface emissivity
                  "evaprate_soil": 0.7,                        #---,       soil surface evaporation rate
                  "lambda_soil": 1.9,                          # W/(m*K),  soil heat conductivity
                  "heatcap_soil": 2.4e6,                       # J/(m^3*K),soil volumetric heat capacity 
                  }
    param.update(param_soil)
    
    
    #%% ASPHALT LAYER PARAMETERS
    param_asphalt = {"asphaltlayer": 1,          #---,       consideration of asphalt layer? 1 = yes, 0 = no
                     "d_asph": 0.18,             # m,        asphalt layer thickness
                     "alpha_asph": 0.93,         #---,       asphalt surface absorptance
                     "epsilon_asph": 0.88,       #---,       asphalt surface emissivity
                     "evaprate_asph": 0.3,       #---,       asphalt surface evaporation rate
                     "lambda_asph": 0.7,         # W/(m*K),  asphalt heat conductivity
                     "heatcap_asph": 1950400    # J/(m^3*K),asphalt volumetric heat capacity
                     }
    
    param.update(param_asphalt)   
      
    
    #%% PIPE PARAMETERS
    param_pipe = {"grid_depth": 1,                  # m,         installation depth beneath surface
                  "lambda_PE": 0.4,                 # W(m*K),    PE heat conductivity
                  "f_fric": 0.025,                  # ---,       pipe friction factor
                  "dp_pipe": 150,                   # Pa/m,      nominal pipe pressure gradient
                  "c_f": 4180,                      # J/(kg*K),  fluid specific heat capacity
                  "rho_f": 1000,                    # kg/m^3,    fluid density                 
                  "conv_pipe": 3600,                # W/(m^2 K), convective heat transfer
                  "R_0": 0.0685,                    # m^2*K/W,   surface correction for thermal resitance (see DIN EN 13941)   
                  }                   
                  
    param.update(param_pipe)  
    
    param_pipe_eco = {"inv_earth_work": 250,                # EUR/m,           preparation costs for pipe installation
                       "inv_pipe_PE": 0.114671,             # EUR/(cm^2*m),    diameter price for PE pipe without insulation                     
                       "pipe_lifetime": 30,                 # a,               pipe life time (VDI 2067)
                       "cost_om_pipe": 0.005                #---,              pipe operation and maintetance costs as share of investment (VDI 2067)
                       }
                
    param.update(param_pipe_eco)
    
    



    
    
    #%% LOAD BALANCING DEVICES PARAMETER
    
    devs = {}

    
    
    #%% BOILER
    devs["BOI"] = {
                   "eta_th": 0.9,       # ---,    thermal efficiency
                   "life_time": 20,     # a,      operation time (VDI 2067)
                   "cost_om": 0.03,     # ---,    annual operation and maintenance costs as share of investment (VDI 2067)
                   }
    
    
    devs["BOI"]["cap_i"] =  {  0: 0,        # MW_th 
                               1: 0.5,      # MW_th
                               2: 5         # MW_th
                               }
    
    devs["BOI"]["inv_i"] = {    0: 0,       # kEUR
                                1: 33.75,   # kEUR
                                2: 96.2     # kEUR
                                }
    

    #%% COMBINED HEAT AND POWER - INTERNAL COMBUSTION ENGINE POWERED BY NATURAL GAS
    devs["CHP"] = {
                   "eta_el": 0.419,     # ---,           electrical efficiency
                   "eta_th": 0.448,     # ---,           thermal efficiency
                   "life_time": 15,     # a,             operation time (VDI 2067)
                   "cost_om": 0.08,     # ---,           annual operation and maintenance costs as share of investment (VDI 2067)
                   }   
    
    devs["CHP"]["cap_i"] =  {  0: 0,        # MW_el
                               1: 0.25,     # MW_el
                               2: 1,        # MW_el
                               3: 3         # MW_el
                               }
    
    devs["CHP"]["inv_i"] = {    0: 0,           # kEUR
                                1: 211.15,      # kEUR
                                2: 410.7,       # kEUR
                                3: 707.6        # kEUR
                                } 
    

    
    #%% ABSORPTION CHILLER
    devs["AC"] = {
                  "eta_th": 0.68,       # ---,        nominal thermal efficiency (cooling power / heating power)
                  "life_time": 18,      # a,          operation time (VDI 2067)
                  "cost_om": 0.03,      # ---,        annual operation and maintenance costs as share of investment (VDI 2067)
                  }
    
    devs["AC"]["cap_i"] =   {  0: 0,        # MW_th
                               1: 0.25,     # MW_th
                               2: 1.535,    # MW_th
                               3: 5.115     # MW_th
                               }
    
    devs["AC"]["inv_i"] = {     0: 0,           # kEUR
                                1: 135.9,       # kEUR
                                2: 366.3,     # kEUR
                                3: 802        # kEUR
                                } 


    


    #%% GROUND SOURCE HEAT PUMP
    
    devs["HP"] = {                  
                  "dT_pinch": 2,                                         # K,    temperature difference between heat exchanger sides at pinch point
                  "dT_min_soil": 2,                                      # K,    minimal temperature difference between soil and brine
                  "life_time": 20,                                       # a,    operation time (VDI 2067)
                  "cost_om": 0.025,                                      #---,   annual operation and maintenance as share of investment (VDI 2067)
                  "dT_evap": 5,                                          # K,    temperature difference of water in evaporator
                  "dT_cond": param["T_hot"] - param["T_cold"],           # K,    temperature difference of water in condenser
                  "eta_compr": 0.8,                                      # ---,  isentropic efficiency of compression
                  "heatloss_compr": 0.1,                                 # ---,  heat loss rate of compression
                  "COP_max": 7,                                          # ---,  maximum heat pump COP
                  }
    
    # Temperatures
    t_c_in = param["T_soil_deep"] - devs["HP"]["dT_min_soil"] + 273.15          # heat source inlet (deep soil temperature - minimal temperature difference)
    dt_c= devs["HP"]["dT_evap"]                                                 # heat source temperature difference
    t_h_in = param["T_cold"] + 273.15                                           # heat sink inlet temperature 
    dt_h = devs["HP"]["dT_cond"]                                                # cooling water temperature difference
 
    # Calculate heat pump COPs
    devs["HP"]["COP"] = calc_COP(devs, param, "HP", [t_c_in, dt_c, t_h_in, dt_h])
    
    # Piece-wise linear cost function
    devs["HP"]["cap_i"] =   {  0: 0,        # MW_th
                               1: 0.2,      # MW_th
                               2: 0.5,
                               3: 4         # MW_th
                               }
    
    devs["HP"]["inv_i"] = {     0: 0,        # kEUR
                                1: 80,      # kEUR
                                2: 150,       # kEUR
                                3: 400
                                } 


    #%% COMPRESSION CHILLER
    
    devs["CC"] = {
                  "life_time": 15,      # a,               operation time (VDI 2067)
                  "cost_om": 0.035,     # ---,             annual operation and maintenance costs as share of investment (VDI 2067)
                  "dT_cond": 5,
                  "dT_evap": param["T_hot"] - param["T_cold"],
                  "dT_min_cooler": 10,
                  "dT_pinch": 2,
                  "eta_compr": 0.75,     # ---,            isentropic efficiency of compression
                  "heatloss_compr": 0.1,  # ---,           heat loss rate of compression 
                  "COP_max": 6
                  }
    
    # Temperatures
    t_c_in = param["T_hot"] + 273.15                            # heat source inlet (equals hot line temperature)
    dt_c = devs["CC"]["dT_evap"]                                # heat source temperature difference
    t_h_in = param["t_air"] + devs["CC"]["dT_min_cooler"] + 273.15       # heat sink inlet temperature (equals temperature of cold cooling water)
    dt_h = devs["CC"]["dT_cond"]                                # cooling water temperature difference
    
    devs["CC"]["COP"] = calc_COP(devs, param, "CC", [t_c_in, dt_c, t_h_in, dt_h])
    
    
    devs["CC"]["cap_i"] = { 0: 0,       # MW_th
                            1: 0.5,     # MW_th
                            2: 4        # MW_th
                            }
    
    
    devs["CC"]["inv_i"] = { 0: 0,      # kEUR
                            1: 95,     # kEUR
                            2: 501     # kEUR
                            } 

    #%% AIR COOLER 
    devs["HYB"] = {
                        "dT_min": 2,
                        "life_time": 20,        # a,        operation time (VDI 2067)
                        "cost_om": 0.035,       #---,       annual operation and maintenance as share of investment (VDI)
                        "inv_var": 240          # kEUR/MW   investment costs (BMVBS)
                        }    
    
    
    #%% ELECTRICAL HEATER

    devs["EH"] = {
                  "eta_th": 0.95,        # ---,             thermal efficiency
                  "life_time": 20,      # a,                operation time
                  "cost_om": 0.01,      # ---,              annual operation and maintenance costs as share of investment
                  "inv_var": 150,       # kEUR/MW           investment costs
                  }

    #%% THERMAL ENERGY STORAGES
    
    for device in ["TES", "CTES"]:
    
        devs[device] = {
                       "min_cap": 0,        # MWh_th,           minimum thermal storage capacity              
                       "sto_loss": 0,   # 1/h,              standby losses over one time step
                       "eta_ch": 1,     # ---,              charging efficiency
                       "eta_dch": 1,    # ---,              discharging efficiency
                       "life_time": 20,     # a,                operation time (VDI 2067 Trinkwasserspeicher)
                       "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment (VDI 2067 Trinkwasserspeicher)
                       "dT_min": 2,
                       }
        
        
        devs[device]["V_i"] = { 0: 0,           # m^3
                                1: 200,         # m^3
                                2: 1000,        # m^3
                                }
        

        devs[device]["inv_i"] = {   0: 0,               # kEUR
                                    1: 128.24,          # kEUR,    
                                    2: 357,             # kEUR       
                                    }

        
    # Storage temperatures
    devs["TES"]["T_min"] = np.max(param["T_hot"]) + devs["TES"]["dT_min"]
    devs["TES"]["T_max"] = 90
    devs["CTES"]["T_min"] = 2
    devs["CTES"]["T_max"] = np.min(param["T_cold"]) - devs["CTES"]["dT_min"]
    
    
    for device in ["TES", "CTES"]: 
        
        devs[device]["cap_i"] = {}
        for i in range(len(devs[device]["V_i"])):
            devs[device]["cap_i"][i] = param["rho_f"] * devs[device]["V_i"][i] * param["c_f"] * (devs[device]["T_max"] - devs[device]["T_min"]) / (1e6 * 3600)
        devs[device]["max_cap"] = devs[device]["cap_i"][len(devs[device]["cap_i"])-1]
        


      
    #%% BATTERY STORAGE
    devs["BAT"] = {
                   "min_cap": 0, 
                   "max_cap": 50,       # MWh_el,           maximum eletrical storage capacity
                   "sto_loss": 0,       # 1/h,              standby losses over one time step
                   "eta_ch": 1,         # ---,              charging efficiency      # 0.9592
                   "eta_dch": 1,        # ---,              discharging efficiency
                   "life_time": 10,     # a,                operation time
                   "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment
                   } 
    
    devs["BAT"]["cap_i"] = { 0: 0,           # MWh_el
                             2: 1,           # MWh_el
                            }
    

    devs["BAT"]["inv_i"] = {   0: 0,         # kEUR
                               1: 1000,      # kEUR,    
                                }

    
    #%% LOAD DOMESCTIC DEVICE PARAMETERS

    devs_dom = {}

    devs_dom["BOI"] = {
                            "life_time": 20,      # a,      operation time (VDI 2067)
                            "inv_var": 67.5,      # EUR/kW, domestic boiler investment costs
                            "inv_fix": 0,         # EUR
                            "cost_om": 0.03,      # ---,    annual operation and maintenance as share of investment (VDI 2067)
                            "eta_th": 0.9         # ---,    boiler thermal efficiency
                            }    
    
    devs_dom["HP"] = {
                            "life_time": 20,       # a,        operation time (VDI 2067)
                            "inv_var": 400,        # EUR/kW,   domestic heat pump investment costs
                            "inv_fix": 0,          # EUR
                            "cost_om": 0.025,      # ---,      annual operation and maintenance as share of investment (VDI 2067)
                            "COP_max": 7,          # ---,      maximum heat pump COP
                            "max_cap_air": 100,     # kW,       maximum heating capacity in case of air-source heat pumps
                            "dT_pinch_air": 8,      # K,        additional pinch point temperature difference in case of air-source heat pump
                            "dT_air": 5             # K,        temperature decrease of air in case of air-source heat pump   
                            }
        
        
    
    devs_dom["CC"] = {
                           "life_time": 15,      # a,      operation time (VDI 2067)
                           "inv_var": 200,       # EUR/kW, domestic heat pump investment costs
                           "inv_fix": 0,         # EUR
                           "cost_om": 0.035,     #---,     annual operation and maintenance as share of investment (VDI 2067)
                           "COP_max": 6,          # ---,    maximum compression chiller COP
                           "dT_pinch_air": 8,     # K,        additional pinch point temperature difference in case of air-sink
                           "dT_air": 5             # K,        temperature increase of air in case of air-source heat pump   
                            }
    
    
    devs_dom["EH"] = {
                            "eta_th": 0.95,        # ---, electric heater efficiency
                            "life_time": 20,       # a,    operation time (TECHNOLOGY DATA FOR ENERGY PLANTS, Danish Energy Agency)
                            "inv_var": 150,        # EUR/kW, domestic heat pump investment costs (162.95)
                            "inv_fix": 0,          # EUR 
                            "cost_om": 0.01,       #---,   annual operation and maintenance as share of investment 
                            }  
    
    devs_dom["FRC"] = {
                            "dT_min": 2,
                            "life_time": 30,       # a,    operation time (VDI 2067)
                            "inv_var": 24.084,     # EUR/kW, domestic heat pump investment costs
                            "inv_fix": 0,
                            "cost_om": 0.03,       #---,   annual operation and maintenance as share of investment 
                            } 
    
    devs_dom["AIR"] = {
                            "dT_min": 10,         # K,       minimum temeprature difference between air and cooling water
                            "life_time": 20,      # a,       operation time (VDI 2067)
                            "inv_var": 40,      # EUR/kW,  domestic heat pump investment costs
                            "inv_fix": 0,         # EUR
                            "cost_om": 0.035,     #---,      annual operation and maintenance as share of investment (VDI)
                            } 
    
    # PV module parameters based on model LG Solar LG360Q1C-A5 NeON R
    # https://www.lg.com/global/business/download/resources/solar/DS_NeONR_60cells.pdf
    devs_dom["PV"] =        {
                            "eta_el_stc": 0.208,        # ---,     electrical efficiency under standard test conditions (STC)
                            "t_cell_stc": 25,           # °C
                            "G_stc": 1000,              # W/m^2
                            "t_cell_noct": 44,          # °C       nominal operation cell temperature (NOCT)
                            "t_air_noct": 20,           # °C,
                            "G_noct": 800,              # W/m^2,       
                            "gamma": -0.003,            # 1/K,
                            "eta_opt": 0.9,             # ---,     optical efficiency according to https://www.homerenergy.com/products/pro/docs/3.11/solar_transmittance.html
                            "life_time": 20,            # a,       operation time (VDI 2067)
                            "inv_var": 900,             # EUR/kW,  PV investment costs   (https://www.photovoltaik4all.de/lg-solar-lg360q1c-a5-neon-r)
                            "inv_fix": 0,               # EUR  
                            "cost_om": 0.02,            #---,      annual operation and maintenance as share of investment (VDI)

                            } 
    
    devs_dom["TES"] =      {
                           "T_max": 90,         # °C,               maximum storage temperature     
                           "T_min": 62,         # °C,               minimal storage temperature      
                           "sto_loss": 0,       # 1/h,              standby losses over one time step
                           "eta_ch": 1,         # ---,              charging efficiency
                           "eta_dch": 1,        # ---,              discharging efficiency
                           "life_time": 20,     # a,                operation time (VDI 2067 Trinkwasserspeicher)
                           "inv_vol": 641.2,    # EUR/m^3           investment costs per m^3 storage volume
                           "inv_fix": 0,        # EUR
                           "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment (VDI 2067 Trinkwasserspeicher)
                           }
    
    
    # Calculate storage investment costs per kWh
    devs_dom["TES"]["inv_var"] = devs_dom["TES"]["inv_vol"] / (param["rho_f"] * param["c_f"] * (devs_dom["TES"]["T_max"] - devs_dom["TES"]["T_min"])) * 1000 * 3600      # EUR/kWh

    # Calculate maximum TES capacity for each building
    devs_dom["TES"]["max_cap"] = {}
    devs_dom["TES"]["min_cap"] = {}    
    for n in nodes:
        devs_dom["TES"]["max_cap"][n] = param["rho_f"] * nodes[n]["V_TES_max"] * param["c_f"] * (devs_dom["TES"]["T_max"] - devs_dom["TES"]["T_min"]) / (1000 * 3600)     # kWh, maximum TES capacity
        devs_dom["TES"]["min_cap"][n] = 0
        
    
    
    # Calculate COP of domestic HPs and CCs
    devs_dom["HP"]["COP"] = {}
    devs_dom["CC"]["COP"] = {}
    
    if param["switch_COP_buildings"]:
        
        for n in nodes:  
            
            # Heat pump Temperatures
            # source
            if not param["switch_stand_alone"]:
                t_c_in = param["T_hot"] + 273.15                                              # heat source inlet (equals hot line temperature)
                dt_c = param["T_hot"] - param["T_cold"]                                       # heat source temperature difference
            else:
                t_c_in = param["t_air"] - devs_dom["HP"]["dT_pinch_air"] + 273.15
                dt_c = devs_dom["HP"]["dT_air"]
            # sink (building)
            t_h_in = nodes[n]["T_heating_return"] + 273.15                                # heat sink inlet temperature (equals return temperature of building heating circle)
            dt_h = nodes[n]["T_heating_supply"] - nodes[n]["T_heating_return"]            # heating circle temperature spread       
            
            # Calculate heat pump COP time series
            devs_dom["HP"]["COP"][n] = calc_COP(devs, param, "HP", [t_c_in, dt_c, t_h_in, dt_h])
            
            
            # Compression chiller temperatures
            # source (building)
            t_c_in = nodes[n]["T_cooling_return"] + 273.15
            dt_c = nodes[n]["T_cooling_return"] - nodes[n]["T_cooling_supply"]
            # sink
            if not param["switch_stand_alone"]:
                t_h_in = param["T_cold"] + 273.15
                dt_h = param["T_hot"] - param["T_cold"]
            else:
                t_h_in = param["t_air"] + devs_dom["CC"]["dT_pinch_air"] + 273.15
                dt_h = devs_dom["HP"]["dT_air"]

            # Calculate compression chiller COP time series
            devs_dom["CC"]["COP"][n] = calc_COP(devs, param, "CC", [t_c_in, dt_c, t_h_in, dt_h])            
                    
    else:   
        devs_dom = calc_COP_buildings(param, devs_dom, nodes)
    
    
    
    # Save maximum PV areas in devs_dom dict
    devs_dom["PV"]["max_area"] = {}
    for n in nodes:
        devs_dom["PV"]["max_area"][n] = nodes[n]["area_PV_max"]
        
    # Calculate PV efficiency time series
    # Cell temperature according to https://www.homerenergy.com/products/pro/docs/3.11/how_homer_calculates_the_pv_cell_temperature.html   
    t_cell = (param["t_air"] + (devs_dom["PV"]["t_cell_noct"] - devs_dom["PV"]["t_air_noct"])*(param["G_sol"]/devs_dom["PV"]["G_noct"])*(1 - (devs_dom["PV"]["eta_el_stc"]*(1-devs_dom["PV"]["gamma"]*devs_dom["PV"]["t_cell_stc"]))/devs_dom["PV"]["eta_opt"])) / (
             (1 + (devs_dom["PV"]["t_cell_noct"] - devs_dom["PV"]["t_air_noct"])*(param["G_sol"]/devs_dom["PV"]["G_noct"])*((devs_dom["PV"]["gamma"]*devs_dom["PV"]["eta_el_stc"])/devs_dom["PV"]["eta_opt"])))    
    devs_dom["PV"]["eta_el"] = devs_dom["PV"]["eta_el_stc"] * (1 + devs_dom["PV"]["gamma"] * (t_cell - devs_dom["PV"]["t_cell_stc"]))
        
    
#%%        
#    #%% BATTERY
#    devs["BAT"] = {"inv_var": 520,      # kEUR/MWh_el,      variable investment
#                   "max_cap": 50,       # MWh_el,           maximum eletrical storage capacity
#                   "sto_loss": 0,       # 1/h,              standby losses over one time step
#                   "eta_ch": 0.9592,    # ---,              charging efficiency
#                   "eta_dch": 0.9592,   # ---,              discharging efficiency
#                   "soc_init": 0.8,     # ---,              maximum initial relative state of charge
#                   "soc_max": 0.8,        # ---,              maximum relative state of charge
#                   "soc_min": 0.2,        # ---,              minimum relative state of charge
#                   "life_time": 10,     # a,                operation time
#                   "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment
#                   }



    

    # Calculate annualized investment of every device
    devs = calc_annual_investment(devs, devs_dom, param)

    return nodes, param, devs, devs_dom





#def get_grid_temp_hot_pipe(mode, time_steps):
#    if mode == "two_point_control":
#        grid_temp_hot_pipe = np.zeros(len(time_steps))
#        grid_temp_summer = 16
#        grid_temp_winter = 22
#        grid_temp_hot_pipe[0:3754] = grid_temp_winter
#        grid_temp_hot_pipe[3754:7040] = grid_temp_summer
#        grid_temp_hot_pipe[7040:8760] = grid_temp_winter
#        
#        with open("D:\\mwi\\Gurobi_Modelle\EctoPlanner\\temp.txt", "w") as outfile:
#            for t in time_steps:
#                outfile.write(str(round(grid_temp_hot_pipe[t],3)) + "\n")   
#        
#    return grid_temp_hot_pipe

#%%
#def calc_pipe_costs(nodes, edges, edge_dict_rev, param):
#    """
#    Calculate variable and fix costs for every edge.
#    """
#    c_fix = {}
#    c_var = {}
#    for e in edges:
#        x1, y1 = nodes[edge_dict_rev[e][0]]["x"], nodes[edge_dict_rev[e][0]]["y"]
#        x2, y2 = nodes[edge_dict_rev[e][1]]["x"], nodes[edge_dict_rev[e][1]]["y"]
#        length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
#        c_fix[e] = param["inv_earth_work"] * length
#        c_var[e] = param["inv_pipe_PE"] * length
#    
#    param["inv_pipe_fix"] = c_fix
#    param["inv_pipe_var"] = c_var
#    return param



#%%
def get_edge_dict(n, nodes):
    compl_graph = nx.complete_graph(n)                                                      # Creates graph with n nodes 0 to n-1 and edges between every pair of nodes
    edge_list = list(compl_graph.edges(data=False))                                         # get list of edges
    edge_dict = {(edge_list[k][0], edge_list[k][1]): k for k in range(len(edge_list))}      # dicts indcluding edge numbers
    edge_dict_rev = {k: (edge_list[k][0], edge_list[k][1]) for k in range(len(edge_list))}
    edges = range(len(edge_list))   # list containing edge indices
    
    # create dict containing edge lengths
    edge_lengths = {}
    for e in edges:
        x1, y1 = nodes[edge_dict_rev[e][0]]["x"], nodes[edge_dict_rev[e][0]]["y"]
        x2, y2 = nodes[edge_dict_rev[e][1]]["x"], nodes[edge_dict_rev[e][1]]["y"]
        length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        edge_lengths[e] = length
                                                           
    return edge_dict, edge_dict_rev, edges, compl_graph, edge_list, edge_lengths



#%%
def transform_coordinates(nodes):
    outProj = Proj(init='epsg:25832')   # ETRS89 / UTM zone 32N
    inProj = Proj(init='epsg:4258')     # Geographic coordinate system: EPSG 4258 (Europe)
    
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
def calc_annual_investment(devs, devs_dom, param):
    """
    Calculation of total investment costs including replacements (based on VDI 2067-1, pages 16-17).

    Parameters
    ----------
    dev : dictionary
        technology parameter
    param : dictionary
        economic parameters

    Returns
    -------
    annualized fix and variable investment
    """

    observation_time = param["observation_time"]
    interest_rate = param["interest_rate"]
    q = 1 + param["interest_rate"]

    
    # Balancing devices
    
    # Calculate capital recovery factor
    CRF = ((q**observation_time)*interest_rate)/((q**observation_time)-1)

    # Calculate annuity factor for each device
    for device in devs.keys():
        
        # Get device life time
        life_time = devs[device]["life_time"]

        # Number of required replacements
        n = int(math.floor(observation_time / life_time))
        
        # Inestment for replcaments
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))

        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))

        # Calculate annualized investments       
        if life_time > observation_time:
            devs[device]["ann_factor"] = (1 - res_value) * CRF 
        else:
            devs[device]["ann_factor"] = ( 1 + invest_replacements - res_value) * CRF 
       
    
    # Building devices
    
    for device in devs_dom.keys():
        
        # Get device life time
        life_time = devs_dom[device]["life_time"]
        inv_fix = devs_dom[device]["inv_fix"]
        inv_var = devs_dom[device]["inv_var"]

        # Number of required replacements
        n = int(math.floor(observation_time / life_time))
        
        # Inestment for replcaments
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))

        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))

        # Calculate annualized investments       
        if life_time > observation_time:
            devs_dom[device]["ann_inv_fix"] = inv_fix * (1 - res_value) * CRF
            devs_dom[device]["ann_inv_var"] = inv_var * (1 - res_value) * CRF
        else:
            devs_dom[device]["ann_inv_fix"] = inv_fix * ( 1 + invest_replacements - res_value) * CRF 
            devs_dom[device]["ann_inv_var"] = inv_var * ( 1 + invest_replacements - res_value) * CRF
            
    
    # Distribution devices (pipes, pumps)
    
    for dev in ["pipe"]: #PUmpe fehlt noch
        
        # Get device life time
        life_time = param[dev + "_lifetime"]
        
        # Number of required replacements
        n = int(math.floor(observation_time / life_time))
        
        # Inestment for replcaments
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))

        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))

        # Calculate annualized investments       
        if life_time > observation_time:
            param["ann_factor_" + dev] = (1 - res_value) * CRF 
        else:
            param["ann_factor_" + dev] = ( 1 + invest_replacements - res_value) * CRF 

    return devs





#%% COP correlation for domestic heat pumps and compression chillers
    
def calc_COP_buildings(param, devs_dom, nodes):
    
    A = 0.67
    B = 12.90
    
    for n in nodes:
        
#        T_sink_HP = np.zeros(8760)
#        for t in range(8760):
#            T_sink_HP[t] = min(nodes[n]["T_heating_supply"][t], devs_dom["HP"]["T_supply_max"])
        T_sink_HP = nodes[n]["T_heating_supply"]
        T_source_HP = param["T_hot"]
        
        T_sink_CC = param["T_hot"]
        T_source_CC = nodes[n]["T_cooling_return"] + 1e-5
        
        
        devs_dom["HP"]["COP"][n] = A * (T_sink_HP + 273.15)/(T_sink_HP - T_source_HP + B)
        devs_dom["CC"]["COP"][n] = A * (T_sink_CC + 273.15)/(T_sink_CC - T_source_CC + B) - 1
        
        for t in range(8760):
            if devs_dom["HP"]["COP"][n][t] > devs_dom["HP"]["COP_max"] or devs_dom["HP"]["COP"][n][t] < 0:
                devs_dom["HP"]["COP"][n][t] = devs_dom["HP"]["COP_max"]
            if devs_dom["CC"]["COP"][n][t] > devs_dom["CC"]["COP_max"] or devs_dom["CC"]["COP"][n][t] < 0:
                devs_dom["CC"]["COP"][n][t] = devs_dom["CC"]["COP_max"] 
    
    return devs_dom
        
        


#%% COP model for ammonia-heat pumps
# Heat pump COP, part 2: Generalized COP estimation of heat pump processes
# DOI: 10.18462/iir.gl.2018.1386
    
def calc_COP(devs, param, device, temperatures):
    
    
    # temperatures: array containing temperature information
    # temperatures = [heat source inlet temperature, heat source temperature difference, heat sink inlet temperature, heat sink temperature difference]
    # each array element can be an array itself or a single value
    # Temperatures must be given in Kelvin !
    

    
    # 0.6 * Carnot
    
#    t_h = 273.15 + param["T_heating_return"] + devs["HP"]["dT_cond"]
#    t_c = 273.15 + param["T_cooling_return"] - devs["HP"]["dT_evap"]
#    
#    COP = 0.6 * t_h / (t_h - t_c)
    
    # get temperature parameters
    t_c_in = temperatures[0]
    dt_c = temperatures[1]
    t_h_in = temperatures[2]
    dt_h = temperatures[3]
    
    
    # device parameters
    dt_pp = devs[device]["dT_pinch"]                # pinch point temperature difference
#    dt_pp = 50
    eta_is = devs[device]["eta_compr"]              # isentropic compression efficiency
    f_Q = devs[device]["heatloss_compr"]            # heat loss rate during compression
    
    # Entropic mean temperautures
    t_h_s = dt_h/np.log((t_h_in + dt_h)/t_h_in)
    t_c_s = dt_c/np.log(t_c_in/(t_c_in - dt_c))
    
    
    if param["switch_clustering"]:
        for d in range(param["n_clusters"]):
            for t in range(24):
                if t_h_s[d][t] == t_c_s[d][t]:
                    t_h_s[d][t] += 1e-5
    else:    
        for t in range(8760):
            if t_h_s[t] == t_c_s[t]:
                t_h_s[t] += 1e-5
    
    #Lorentz-COP
    COP_Lor = t_h_s/(t_h_s - t_c_s)
    
    
    # linear model equations
    dt_r_H = 0.2*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) + 0.2*dt_h + 0.016        # mean entropic heat difference in condenser deducting dt_pp
    w_is = 0.0014*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) - 0.0015*dt_h + 0.039    # ratio of isentropic expansion work to isentropic compression work
    
    
    # help values
    num = 1 + (dt_r_H + dt_pp)/t_h_s
    denom = 1 + (dt_r_H + 0.5*dt_c + 2*dt_pp)/(t_h_s - t_c_s)
    
    # COP
    COP = COP_Lor * num/denom * eta_is * (1 - w_is) + 1 - eta_is - f_Q

    if device == "CC":
        COP = COP - 1   # consider COP definition for compression chillers (COP_CC = Q_0/P_el = (Q - P_el)/P_el = COP_HP - 1)
    
    # limit COP's
    COP_max = devs[device]["COP_max"]
    
    if param["switch_clustering"]:
        for d in range(param["n_clusters"]):
            for t in range(24):
                if COP[d,t] > COP_max or COP[d,t] < 0:
                    COP[d,t] = COP_max
    else:
        for t in range(len(COP)):
            if COP[t] > COP_max or COP[t] < 0:
                COP[t] = COP_max


                

    return COP

