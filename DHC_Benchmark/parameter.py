# -*- coding: utf-8 -*-
"""

Author: Marco Wirtz, Institute for Energy Efficient Buildings and Indoor Climate, RWTH Aachen University, Germany

Created: 01.09.2018

"""

import numpy as np
import math
#import sun
import os

import grid
import soil
import pylab as plt



def load_params():
    """
    Returns all known data for optmization model.
    """
  
    #%% GENERAL PARAMETERS
    param = {"interest_rate":  0.05,        # ---,          interest rate
             "observation_time": 20.0,      # a,            project lifetime
             "price_gas": 0.02824,          # kEUR/MWh,     natural gas price
             "price_cap_gas": 12.149,       # kEUR/(MW*a)   capacity charge for gas grid usage
             "price_el": 0.14506,           # kEUR/MWh,     electricity price
             "price_cap_el": 59.660,        # kEUR/(MW*a)   capacity charge for electricity grid usage
#             "self_charge": 0.0272,         # kEUR/MWh      charge on on-site consumption of CHP-generated power   
             "revenue_feed_in": 0.06442,    # kEUR/MWh,     feed-in revenue for CHP-generated power
             "gas_CO2_emission": 0.2,       # t_CO2/MWh,    specific CO2 emissions (natural gas)
             "grid_CO2_emission": 0.503,    # t_CO2/MWh,    specific CO2 emissions (grid)
#             "pv_stc_area": 10000,         # m2,           roof area for pv or stc
             "MIPGap":      0.0001          # ---,          MIP gap
             }
    
    #%% SWITCHES
    
    param_switches = {"switch_low_temp": 1,             # 1: low-temperature heating 0: heating temperature according to real data from FZ Jülich
                      "switch_n_1": 0,                  # consider n-1 redundancy of heating and cooling generation
                      "switch_hp": 1,                    # consider heat pump between cooling and heating return pipes?
                      "switch_transient_hp": 0,          # consider variable heat pump outlet temperature
                      "switch_pump": 0                   # consider pumping costs
                      }
    
    param.update(param_switches)


    #%% SOIL PARAMETERS   
    param_soil = {"alpha_soil": 0.8,            #---,       soil surface absorptance
                  "epsilon_soil": 0.9,          #---,       soil surface emissivity
                  "evaprate_soil": 0.7,         #---,       soil surface evaporation rate
                  "lambda_soil": 1.9,           # W/(m*K),  soil heat conductivity
                  "heatcap_soil": 2.4e6,        # J/(m^3*K),soil volumetric heat capacity 
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
    param_pipe = {"grid_depth": 1,                  # m,       installation depth beneath surface
                  "lambda_ins": 0.03,              # W/(m*K),  insulation heat conductivity (set pipes GmbH)
                  "lambda_PE": 0.4,                 # W(m*K),  PE heat conductivity (set pipes GmbH, wikipedia)
                  "lambda_steel": 50,               # W(m*K),  steel heat conductivity (set pipes GmbH)
                  "R_0": 0.0685,                    # m^2*K/W, pipe surface correction for thermal resitance (see DIN EN 13941)   
                  "f_fric": 0.025,                  # ---,     pipe friction factor
                  "dp_pipe": 150,                   # Pa/m,    nominal pipe pressure gradient
                  "c_f": 4180,                      # J/(kg*K),fluid specific heat capacity
                  "rho_f": 1000,                    # kg/m^3,  fluid density
#                  "t_soil": 0.6,                   # m,       thickness of soil layer around the pipe to calculate heat transfer into ground
                  "conv_pipe": 3600,                 # W/(m^2*K) convective heat transfer coefficient (VDI Wärmeatlas, v = 1 m/s)
                  "dp_single": 0.3,                 # ---,      single pressure drops as part of total pipe pressure loss
                  "dp_substation": 1e5,             # Pa/MW     pressure drop at substation
                              
                  }
    
    param.update(param_pipe)  
                
    param_pipe_eco = {"inv_ground": 250,                 # EUR/m,        costs for pipe installment
                       "inv_pipe_isolated": 2553.44,     # EUR/m^2,      diameter price for isolated pipes
                       "inv_pipe_isolated_fix": 18.27,   # EUR/m         fix price for isolated pipes
                       "inv_pipe_PE": 1146.71,           # EUR/(m^2*m),  diameter price for PE pipe without insulation
                       "pipe_lifetime": 30,              # a,            pipe life time (VDI 2067)
                       "cost_om_pipe": 0.005              #---,           pipe operation and maintetance costs as share of investment (VDI 2067)
                       }
                
    
    param.update(param_pipe_eco)
    
    #%% PUMP PARAMETERS
    
    param_pump ={ "eta_pump": 0.7,                  # ---,        pump electrical efficiency
                  "inv_pump": 500,                  # kEUR/MW,    pump investment costs
                  "pump_lifetime": 10,              # a,          pump life time (VDI 2067)
                  "cost_om_pump": 0.03,             # ---,        pump operation and maintenance costs (VDI 2067)
                  "pump_electricity_costs": 0.05    # kEUR/MWh    pump electricity costs (equals representative CHP power generation costs)
                 }
    
    param.update(param_pump)
    
    
    #%% SUBSTATION PARAMETERS
    
    param_substation = {"inv_sub_fix": 3.896,              # kEUR,         fix investment costs for substations
                        "inv_sub_var": 11.436,            # kEUR/MW       variable investment costs for substations
                        "sub_lifetime": 30,              # a,            substation lifetime (VDI 2067)
                        "cost_om_sub": 0.03              # ---,          substation operation and maintenance costs as share of investment
                        }
    
    param.update(param_substation)
    
    
    #%% TEMPERATURES
    if not param["switch_low_temp"]:
        T_heating_return = 70
    else:
        T_heating_return = 30
    
    param_temperatures = {"T_heating_supply_low": 60,                # °C,   supply temperature in case of low-temperature heating network
                          "T_heating_return": T_heating_return,      # °C,   heating return temperature
                          "T_cooling_supply": 6,                     # °C,   cooling supply temperature
                          "T_cooling_return": 12}                    # °C,   cooling return temperature

    
    param.update(param_temperatures)
    
    
    #%% GRID SIZING
    # design grid properties for the given input data and parameters
    grid_data, param = grid.design_grid(param)
    #grid.plotGrid()


    
    
     #%% LOADS

    dem = {} 
    

    dem_buildings = grid.load_demands(grid_data)
        
    dem["heat"] = dem_buildings["heating"]["sum"]      # MW, heating demand of all buildings
    dem["cool"] = dem_buildings["cooling"]["sum"]      # MW, cooling demand of all buildings   
    
    param["max_heating"] = max(dem["heat"])
    param["max_cooling"] = max(dem["cool"])
    
    param["load_factor_heating"] = sum(dem["heat"])/param["max_heating"]/8760
    param["load_factor_cooling"] = sum(dem["cool"])/param["max_cooling"]/8760
    
#    print(param["load_factor_heating"])
#    print(param["load_factor_cooling"])
    
#    print(sum(dem["heat"][t] for t in range(8760)))
#    print(sum(dem["cool"][t] for t in range(8760)))

#    print(param["max_heating"])
#    print(param["max_cooling"])
    
#    Q_Nenn = np.max(dem["heat"])
#    Q_Jahr = np.sum(dem["heat"])
#    Q0_Nenn = np.max(dem["cool"])
#    Q0_Jahr = np.sum(dem["cool"])   

    #%% PUMP SIZING
    
    param = grid.design_pump(grid_data, param, dem_buildings)

    # Electrical energy for pumping
    param["pump_energy"] = (param["load_factor_heating"] * param["pump_cap_heating"] + param["load_factor_cooling"] * param["pump_cap_cooling"])* 8760      # MWh, electrical pump energy
    
#    print(param["pump_energy"])
 
    
#%% THERMAL LOSSES
   
    # calculate heating and cooling losses of the grid
    losses = soil.calculateLosses(param, grid_data)
    
    print(sum(losses["cooling_grid"][t] for t in range(8760)))
          
    # Add losses to building demands to get total grid demand
    dem["heat"] = dem["heat"] + losses["heating_grid"]
    dem["cool"] = dem["cool"] + losses["cooling_grid"]
  

#    print(max(dem["heat"][t] for t in range(8760)))    
#    print(max(dem["cool"][t] for t in range(8760))) 
    
#    time_max = np.argmax(dem["heat"])
#    time_max2 = np.argmax(dem["cool"])
    
#    anteil = sum(losses["heating_grid"])/sum(dem["heat"])
#    anteil2 = sum(losses["cooling_grid"])/sum(dem["cool"])
#    print("Anteil Wärmeverluste am Gesamtbedarf = " + str(anteil))
#    print("Anteil Kälteverluste am Gesamtbedarf = " + str(anteil2))
    
    
#%%   
    # Improve numeric by deleting very small loads
    eps = 0.01 # MW
    for load in ["heat", "cool"]:
        for k in range(len(dem[load])):
           if dem[load][k] < eps:
              dem[load][k] = 0
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    devs = {}

    #%% BOILER
    devs["BOI"] = {
                   "eta_th": 0.9,       # ---,    thermal efficiency
                   "life_time": 20,     # a,      operation time (VDI 2067)
                   "cost_om": 0.03,     # ---,    annual operation and maintenance costs as share of investment (VDI 2067)
                   }
    
    
    # Piece-wise linear cost function
    devs["BOI"]["cap_i"] =  {  0: 0,        # MW_th 
                               1: 0.5,      # MW_th
                               2: 5         # MW_th
                               }
    
    devs["BOI"]["inv_i"] = {    0: 0,       # kEUR
                                1: 33.75,   # kEUR
                                2: 96.2     # kEUR
                                }

    #%% COMBINED HEAT AND POWER
    devs["CHP"] = {
                   "eta_el": 0.419,     # ---,           electrical efficiency
                   "eta_th": 0.448,     # ---,           thermal efficiency
                   "life_time": 15,     # a,             operation time (VDI 2067)
                   "cost_om": 0.08,     # ---,           annual operation and maintenance costs as share of investment (VDI 2067)
                   }   
    
    # Piece-wise linear cost function
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
    

    #%% WATER SOURCE HEAT PUMP
    devs["HP"] = {
                  "dT_pinch": 2,           # K,    temperature difference between heat exchanger sides at pinch point
                  "life_time": 20,         # a,    operation time (VDI 2067)
                  "cost_om": 0.025,        #---,   annual operation and maintenance as share of investment (VDI 2067)
                  "dT_evap": 6,            # K,    temperature difference of water in evaporator
                  "dT_cond": 20,           # K,    temperature difference of water in condenser
                  "eta_compr": 0.8,        # ---,  isentropic efficiency of compression
                  "heatloss_compr": 0.1,   # ---,  heat loss rate of compression
                  "COP_max": 7,            # ---,  maximum heat pump COP
                  "lin_error": 0.001       # ---, tolerance for COP-linearization error
                  }
 
    # Calculate heat pump COP
    if param["switch_transient_hp"] == 0:
        devs["HP"]["COP"] = calc_COP(devs, param, "HP",devs["HP"]["dT_cond"])
    else:
        devs = lin_COP_HP(param, devs)
    
    # Piece-wise linear cost function
    devs["HP"]["cap_i"] =   {  0: 0,        # MW_th
                               1: 0.5,      # MW_th
                               2: 4         # MW_th
                               }
    
    devs["HP"]["inv_i"] = {     0: 0,        # kEUR
                                1: 100,      # kEUR
                                2: 400       # kEUR
                                } 
    
    

    #%% ABSORPTION CHILLER
    devs["AC"] = {
                  "eta_th": 0.68,       # ---,        nominal thermal efficiency (cooling power / heating power)
                  "life_time": 18,      # a,          operation time (VDI 2067)
                  "cost_om": 0.03,      # ---,        annual operation and maintenance costs as share of investment (VDI 2067)
                  }
    
    # Piece-wise linear cost function
    devs["AC"]["cap_i"] =   {  0: 0,        # MW_th
                               1: 0.25,     # MW_th
                               2: 1.535,    # MW_th
                               3: 5.115     # MW_th
                               }
    
    devs["AC"]["inv_i"] = {     0: 0,         # kEUR
                                1: 135.9,     # kEUR
                                2: 366.3,     # kEUR
                                3: 802        # kEUR
                                } 

    #%% COMPRESSION CHILLER
    devs["CC"] = {
                  "life_time": 15,          # a,               operation time (VDI 2067)
                  "cost_om": 0.035,         # ---,             annual operation and maintenance costs as share of investment (VDI 2067)
                  "dT_cond": 5,             # K,               temperature difference of water in condenser
                  "dT_min_cooler": 10,      # K,               minimal temperature difference between cooling water and air 
                  "dT_pinch": 2,            # K,               temperature difference between heat exchanger sides at pinch point 
                  "eta_compr": 0.75,        # ---,             isentropic efficiency of compression
                  "heatloss_compr": 0.1,    # ---,             heat loss rate of compression 
                  "COP_max": 6              # ---,             maximum compression chiller COP
                  }
    
    # In the heat pump system the CC COP is calculated for the worst case in which the heat pump supplies all cooling load
    if param["switch_hp"]:
        devs["CC"]["dT_evap"] = 1e-5
    else:
        devs["CC"]["dT_evap"] = param["T_cooling_return"] - param["T_cooling_supply"]
    
    
    # Calculate compression chiller COPs 
    devs["CC"]["COP"] = calc_COP(devs, param, "CC", devs["CC"]["dT_cond"])
    
    
    # Piece-wise linear cost function
    devs["CC"]["cap_i"] = { 0: 0,       # MW_th
                            1: 0.5,     # MW_th
                            2: 4        # MW_th
                            }
    
    
    devs["CC"]["inv_i"] = { 0: 0,       # kEUR
                            1: 95,      # kEUR
                            2: 501      # kEUR
                            } 
    
    
    #%% (HEAT) THERMAL ENERGY STORAGE
#    devs["TES"] = {
#                   "switch_TES": 0,     # toggle availability of thermal storage
#                   "max_cap": 250,      # MWh_th,          maximum thermal storage capacity
#                   "min_cap": 0,        # MWh_th,           minimum thermal storage capacity              
#                   "sto_loss": 0.005,   # 1/h,              standby losses over one time step
#                   "eta_ch": 0.975,     # ---,              charging efficiency
#                   "eta_dch": 0.975,    # ---,              discharging efficiency
#                   "max_ch": 1000,      # MW,               maximum charging power
#                   "max_dch": 1000,     # MW,               maximum discharging power
#                   "soc_init": 0.8,     # ---,              maximum initial state of charge
#                   "soc_max": 1,        # ---,              maximum state of charge
#                   "soc_min": 0,        # ---,              minimum state of charge
#                   "life_time": 20,     # a,                operation time (VDI 2067 Trinkwasserspeicher)
#                   "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment (VDI 2067 Trinkwasserspeicher)
#                  
#                   }
#    
#    devs["TES"]["cap_i"] =   { 0: 0,         # MWh_th,      depends on temperature difference! Q = V * c_p * rho * dT
#                               1: 8.128,     # MWh_th
#                               2: 40.639,    # MWh_th
#                               3: 243.833    # MWh_th
#                               }
#    
#    devs["TES"]["inv_i"] = {    0: 0,              # kEUR
#                                1: 147.2,          # kEUR,    includes factor of 1.15 for pressure correction factor due to high temperatures; higher pressure is needed to prevent evaporation
#                                2: 410.55,         # kEUR
#                                3: 1083.3          # kEUR
#                                } 
    
 #%%   
    # Calculate annuity factor of every device and annualized costs for energy distribution
    devs, param = calc_annual_investment(devs, param, grid_data, dem_buildings)  
    
#    print("Trassenlänge: " + str(param["length_pipes"]) +  " m")
#    print("Leistungsdichte Wärmenetz: " + str(Q_Nenn/param["length_pipes"]*1000) + " MW/km  /  " + str(Q_Jahr/param["length_pipes"]*1000) + " MWh/(km*a)")
#    print("Leistungsdichte Kältenetz: " + str(Q0_Nenn/param["length_pipes"]*1000) + " MW/km  /  " + str(Q0_Jahr/param["length_pipes"]*1000) + " MWh/(km*a)")
#    
    return (devs, param, dem)

#%%
#def get_irrad_profile(ele, azim, weather_dict):
#    """
#    Calculates global irradiance on tilted surface from weather file.
#    """
#
#    # Load time series as numpy array
#    dtype = dict(names = ['id','data'], formats = ['f8','f8'])
#    sun_diffuse = np.array(list(weather_dict["Diffuse Horizontal Radiation"].items()), dtype=dtype)['data']
#    sun_global = np.array(list(weather_dict["Global Horizontal Radiation"].items()), dtype=dtype)['data']
#    sun_direct = sun_global - sun_diffuse
#
#    # Define local properties
#    time_zone = 7                # ---,      time zone (weather file works properly with time_zone = 7, although time_zone = 8 is proposed in the weather file)
#    location = (31.17, 121.43)   # degree,   latitude, longitude of location
#    altitude = 7.0               # m,        height of location above sea level
#
#    # Calculate geometric relations
#    geometry = sun.getGeometry(0, 3600, 8760, time_zone, location, altitude)
#    (omega, delta, thetaZ, airmass, Gon) = geometry
#    theta = sun.getIncidenceAngle(ele, azim, location[0], omega, delta)
#
#    theta = theta[1] # cos(theta) is not required
#
#    # Calculate radiation on tilted surface
#    return sun.getTotalRadiationTiltedSurface(theta, thetaZ, sun_direct, sun_diffuse, airmass, Gon, ele, 0.2)

#%%
#def calc_pv(dev, weather_dict):
#    """
#    Calculates photovoltaic output in MW per MW_peak.
#    Model based on http://www.sciencedirect.com/science/article/pii/S1876610213000829, equation 5.
#
#    """
#
#    # Calculate global tilted irradiance in W/m2
#    gti_pv = get_irrad_profile(dev["elevation"], dev["azimuth"], weather_dict)
#
#    # Get ambient temperature from weather dict
#    temp_amb = np.array(list(weather_dict["Dry Bulb Temperature"].items()), dtype=dict(names = ['id','data'], formats = ['f8','f8']))['data']
#
#    temp_cell = temp_amb + gti_pv / dev["solar_noct"] * (dev["temp_cell_noct"] - temp_amb)
#    eta_noct = dev["power_noct"] / (dev["module_area"] * dev["solar_noct"])
#    eta_cell = eta_noct * (1 - dev["gamma"] * (temp_cell - dev["temp_amb_noct"]))
#
#    # Calculate collector area (m2) per installed capacity (MW_peak)
#    area_per_MW_peak = dev["module_area"] / (dev["nom_module_power"] / 1000000)
#
#    # Calculate power generation in MW/MW_peak
#    pv_output = eta_cell * (gti_pv / 1000000) * area_per_MW_peak
#
#    return dict(enumerate(pv_output))

#%%
#def calc_stc(devs, weather_dict):
#    """
#    Calculation of thermal output in MW/MW_peak according to ISO 9806 standard (p. 43).
#
#    """
#
#    dev = devs["STC"]
#
#    # Calculate global tilted irradiance in W/m2
#    gti_stc = get_irrad_profile(dev["elevation"], dev["azimuth"], weather_dict)
#
#    # Get ambient temperature from weather dict
#    temp_amb = np.array(list(weather_dict["Dry Bulb Temperature"].items()), dtype=dict(names = ['id','data'], formats = ['f8','f8']))['data']
#
#    # Calculate heat output in W/m2
#    stc_output_m2 = np.zeros(gti_stc.size)
#    t_norm = (dev["temp_mean"] - temp_amb) / gti_stc
#    eta_th = dev["eta_0"] - dev["a1"] * t_norm - dev["a2"] * t_norm**2 #* gti_stc
#    for t in range(eta_th.size):
#        if not np.isfinite(eta_th[t]):
#            eta_th[t] = 0
#        stc_output_m2[t] = max(eta_th[t] * gti_stc[t], 0)
#
#    # Calculate collector area (m2) per installed capacity (MW_peak)
#    area_per_MW_peak = 1000000 / dev["power_per_m2"]
#
#    # Calculate thermal heat output in MW/MW_peak
#    stc_output = stc_output_m2 * area_per_MW_peak / 1000000
#
#    return dict(enumerate(stc_output))

#%%
#def calc_wind(dev, weather_dict):
#    """
#    Calculation power output from wind turbines in MW/MW_peak.
#    
#    """
#    
#    power_curve = dev["power_curve"]
#    
#    dev["power"] = {}
#    for t in range(len(weather_dict["Wind Speed"])):
#        wind_speed_ground = weather_dict["Wind Speed"][t]
#        wind_speed_shaft = wind_speed_ground * (dev["hub_height"] / dev["ref_height"]) ** dev["expo_a"]
#        
#        # if cases can then be eliminated, if np.interp is used
#        if wind_speed_shaft <= 0:
#            dev["power"][t] = 0
#        elif wind_speed_shaft > power_curve[len(power_curve)-1][0]:
#            print("Warning: Wind speed is " + str(wind_speed_shaft) + " m/s and exceeds wind power curve table.")
#            dev["power"][t] = 0
#    
#        # Linear interpolation
#        
#        # better use: #    res = np.interp(2.5, speed_points, power_points)
#        # do not use this extra function calc_wind, move it directly to wind data section
#        
#        else:
#            for k in range(len(power_curve)):
#                if power_curve[k][0] > wind_speed_shaft:
#                    dev["power"][t] = (power_curve[k][1]-power_curve[k-1][1])/(power_curve[k][0]-power_curve[k-1][0]) * (wind_speed_shaft-power_curve[k-1][0]) + power_curve[k-1][1]
#                    break
#            
#    return dev

#%%
#def calc_COP_AHSP(devs, weather_dict):
#    """
#    Calculation of COP for air source heat pump based on carnot efficiency.
#
#    """
#
#    devs["ASHP"]["COP"] = {}
#    for t in weather_dict["Dry Bulb Temperature"].keys():
#        air_temp = weather_dict["Dry Bulb Temperature"][t]
#        devs["ASHP"]["COP"][t] = devs["ASHP"]["eta"] * (devs["ASHP"]["t_supply"]/(devs["ASHP"]["t_supply"]-(air_temp + 273)))
#    return devs["ASHP"]["COP"]

#%%
def calc_annual_investment(devs, param, grid_data, dem_buildings):
    """
    Calculation of total investment costs including replacements and residual value (based on VDI 2067-1, pages 16-17).
    
    Annuity factors of generation devices are returned.
    Total annualized costs of distribution devices are returned.
    
    """

    observation_time = param["observation_time"]
    interest_rate = param["interest_rate"]
    q = 1 + param["interest_rate"]

    # Calculate capital recovery factor
    CRF = ((q**observation_time)*interest_rate)/((q**observation_time)-1)

 
    # Calculate annuity factor for generation devices
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
            
      
    

    # Pipe costs
    
    length = 0
    inv_pipes = 0
       
    life_time = param["pipe_lifetime"]

    # Sum up investment costs for each edge
    for item in grid_data["edges"]:
        length = length + grid_data["edges"][item]["length"]
        d_h = grid_data["edges"][item]["diameter_heating"]
        d_c = grid_data["edges"][item]["diameter_cooling"]
        # binary variables to check if a pipe is needed at this edge (some buildings don't have cooling demand)
        x_h = d_h > 0
        x_c = d_c > 0 
        inv_pipes = inv_pipes + ((x_h or x_c) * 0.5 * param["inv_ground"] + x_h * (param["inv_pipe_isolated_fix"] + param["inv_pipe_isolated"] * d_h**2) + x_c * param["inv_pipe_PE"] * d_c**2) * 2 * grid_data["edges"][item]["length"]
      
    # Number of required replacements
    n = int(math.floor(observation_time / life_time))       
    # Inestment for replcaments
    invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))
    # Residual value of final replacement
    res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))
    
    if life_time > observation_time:
        param["tac_pipes"] = (CRF * inv_pipes * (1 - res_value) + param["cost_om_pipe"] * inv_pipes) / 1000      # kEUR,     annualized pipe costs
    else:
        param["tac_pipes"] = (CRF * inv_pipes * (1 + invest_replacements - res_value) + param["cost_om_pipe"] * inv_pipes) / 1000
        
    param["length_pipes"] = length                                                                           # m,        one-way length of heating network


    

    # substation costs
    
    life_time = param["sub_lifetime"]
    max_loads = []
    for item in dem_buildings["heating"]:
        if item != "sum":
            max_loads.append(np.max(dem_buildings["heating"][item]))
            max_loads.append(np.max(dem_buildings["cooling"][item]))
                                  
    inv_subs = sum((max_loads[k] > 0) * param["inv_sub_fix"] + param["inv_sub_var"] * max_loads[k] for k in range(len(max_loads)))
    
    # Number of required replacements
    n = int(math.floor(observation_time / life_time))       
    # Inestment for replcaments
    invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))
    # Residual value of final replacement
    res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))
     
    if life_time > observation_time:
        param["tac_subs"] = (CRF * inv_subs * (1 - res_value) + param["cost_om_sub"] * inv_subs)      # kEUR,     annualized substation costs
    else:
        param["tac_subs"] = (CRF * inv_subs * (1 + invest_replacements - res_value) + param["cost_om_sub"] * inv_subs)
    

    
    
    # Pump investment and operation/maintenance costs
    
    if param["switch_pump"]:
    
        life_time = param["pump_lifetime"]
                                      
        inv_pump = param["inv_pump"] * (param["pump_cap_heating"] + param["pump_cap_cooling"])  #kEUR, pump investment costs
        
        # Number of required replacements
        n = int(math.floor(observation_time / life_time))       
        # Inestment for replcaments
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))
        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))
         
        if life_time > observation_time:
            param["tac_pump"] = (CRF * inv_pump * (1 - res_value) + param["cost_om_pump"] * inv_pump)      # kEUR,     annualized pump investment and maintenance costs
        else:
            param["tac_pump"] = (CRF * inv_pump * (1 + invest_replacements - res_value) + param["cost_om_pump"] * inv_pump) 
    
                
        # add Pump electricity costs
        param["pump_costs_el"] = param["pump_energy"] * param["pump_electricity_costs"]
        param["tac_pump"] += param["pump_costs_el"]
    
    else:
    
        param["tac_pump"] = 0
    
    
    #print("pumpe Strom" + str(param["tac_pump"]))
    
    
    # Sum up distribution costs
    param["tac_distr"] = param["tac_pipes"] + param["tac_subs"]  + param["tac_pump"]
    
    
#    print(param["tac_distr"])
                                                                           

    return devs, param



#%% COP model for ammonia-heat pumps
# Heat pump COP, part 2: Generalized COP estimation of heat pump processes
# DOI: 10.18462/iir.gl.2018.1386
    
def calc_COP(devs, param, device, dt_h):
    
    # 0.6 * Carnot
    
#    t_h = 273.15 + param["T_heating_return"] + devs["HP"]["dT_cond"] + devs["HP"]["dT_pinch"]
#    t_c = 273.15 + param["T_cooling_return"] - devs["HP"]["dT_evap"] - devs["HP"]["dT_pinch"]
#    
#    COP = 0.6 * t_h / (t_h - t_c)
    
    
    

    # get temperature parameters
#    dt_h = devs[device]["dT_cond"]                  # heat sink temperature difference
    dt_c = devs[device]["dT_evap"]                  # heat sourve temperature difference
    
    if device == "HP":
        t_h_in = param["T_heating_return"] + 273.15   # heat sink inlet temperature
        t_c_in = param["T_cooling_return"] + 273.15   # heat source inlet temperature
    else:
        t_air = np.loadtxt(open("input_data/weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(0))               # Air temperature °C
        t_h_in = t_air + devs["CC"]["dT_min_cooler"] + 273.15                                                             # heat sink inlet temperature (cold cooling water)
        if param["switch_hp"]:
            t_c_in = param["T_cooling_supply"] + 273.15                                                                   # heat source inlet temperature
        else:
            t_c_in = param["T_cooling_return"] + 273.15
            
    # Modeling parameters
    dt_pp = devs[device]["dT_pinch"]                # pinch point temperature difference
    eta_is = devs[device]["eta_compr"]              # isentropic compression efficiency
    f_Q = devs[device]["heatloss_compr"]            # heat loss rate during compression
    
    # Entropic mean temperautures
    t_h_s = dt_h/np.log((t_h_in + dt_h)/t_h_in)
    t_c_s = dt_c/np.log(t_c_in/(t_c_in - dt_c))
    
    #Lorentz-COP
    COP_Lor = t_h_s/(t_h_s - t_c_s)
    
    
    # linear model equations
    dt_r_H = 0.2*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) + 0.2*dt_h + 0.016        # mean entropic heat difference in condenser deducting dt_pp
    w_is = 0.0014*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) - 0.0015*dt_h + 0.039    # ratio of isentropic expansion work and isentropic compression work
    
    
    # help values
    num = 1 + (dt_r_H + dt_pp)/t_h_s
    denom = 1 + (dt_r_H + 0.5*dt_c + 2*dt_pp)/(t_h_s - t_c_s)
    
    # COP
    COP = COP_Lor * num/denom * eta_is * (1 - w_is) + 1 - eta_is - f_Q
    
    if device == "CC":
        COP = COP - 1   # consider COP definition for compression chillers  (COP_CC = Q_0/P_el = (Q - P_el)/P_el = COP_HP - 1)
 
    
    # limit COP's
    COP_max = devs[device]["COP_max"]
    
    if device == "CC":
        for t in range(len(COP)):
            if COP[t] > COP_max:
                COP[t] = COP_max
                
#        plt.plot(t_air,COP,".")
#        plt.show()
#        
#        plt.xlabel("Lufttemperatur °C")
#        plt.ylabel("COP_CC")
    
    elif device == "HP":
        if COP > COP_max:
            COP = COP_max
#        print(COP)
    
#        print(COP)
    return COP

    
    


#%%
# COP linearization in case of variable heat pump supply temperatures
def lin_COP_HP(param, devs):
    
#    t_supply_max = np.max(param["T_heating_supply"])
    dt_h_max = devs["HP"]["dT_cond"]
    
    n_values = 500
    dt_h = np.linspace(0,dt_h_max,n_values)
    
    
    COP = np.zeros(n_values)    
    eps = 1e-5
    
    # Calculate COP-Values
    for k in range(n_values):
        COP[k] = calc_COP(devs, param, "HP", dt_h[k]+eps)  
        
    
    
    ### Linearize ratio := dt_H/COP   
    N = 2    
    for k in range(1000):

        ratio_lin = np.zeros(n_values)
                   
        dt_h_i = np.zeros(N)
        ratio_i = np.zeros(N)
        
        for i in range(N):
            
            # Calculate support points
            index = int( i/(N-1) * (n_values -1) )
            dt_h_i[i] = dt_h[index]
            COP_i = COP[index]            
            ratio_i[i] = dt_h_i[i] / COP_i
            
            # Caclulate linearized values            
            if i >= 1:
                index_start = int( (i-1)/(N-1) * (n_values - 1) )
                ratio_i_start = ratio_i[i-1]
                
                index_end = index
                ratio_i_end = ratio_i[i]
                
                for index in np.arange(index_start + 1, index_end + 1):
                    ratio_lin[index] = ratio_i_start + (index - index_start)/(index_end - index_start) * (ratio_i_end - ratio_i_start)
           
            
        # Calculate deviation between exact values and liearized values
        ratio = dt_h / COP
        errors = abs(ratio_lin - ratio)
        
        # if errors are too high, increase number of support points
        next_N = 0
        for e in errors:
            if e > devs["HP"]["lin_error"]:
                next_N = 1
        
        if next_N:        
            N += 1                
        else:
            break
        
    
    print(str(N) + " Stützstellen")
    plt.plot(dt_h, dt_h/COP)
    plt.plot(dt_h_i, ratio_i)
    plt.plot(dt_h, ratio_lin)
        
    plt.show()
    
    print(dt_h_i)
    print(ratio_i)
    
    # add support points to devs
    devs["HP"]["dt_h_i"] = {}
    devs["HP"]["ratio_i"] = {}
    
    for i in range(N):
        devs["HP"]["dt_h_i"][i] = dt_h_i[i]
        devs["HP"]["ratio_i"][i] = ratio_i[i]

    return devs

        
