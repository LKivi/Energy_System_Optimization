# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:13:13 2018

@author: lkivi
"""

import numpy as np
from scipy.optimize import leastsq
import cmath
import grid



#%%
# Calculate thermal losses in heating grid and cooling grid
# Thermal loss calculation is based on DIN EN 13941; thermal interaction between supply pipe and return pipe is neglected
def calculateLosses(param, data):
    
    
    T_soil = calculateSoilTemperature(param)
    
    losses = {}

    #%% HEATING GRID LOSSES  
    
    losses["heating_grid"] = np.zeros(8760)
    path = "input_data/pipes_heating.txt"
    # available inner pipe diameters for the heating network
    diameters = param["diameters"]["heating"]
    t_steel = np.loadtxt(open(path, "rb"), delimiter = ",", usecols=(1))        # m,  available steel pipe thicknesses
    t_ins = np.loadtxt(open(path, "rb"), delimiter = ",", usecols=(2))          # m,  available insulation thicknesses
    t_PE = np.loadtxt(open(path, "rb"), delimiter = ",", usecols=(3))           # m,  available PE coat thicknesses
    
    # create dictionary for heating pipe geometry
    pipes = {}
    for i in range(np.size(diameters)):
        pipes[diameters[i]] = {"t_steel": t_steel[i],
                               "t_ins": t_ins[i],
                               "t_PE": t_PE[i]
                               }
    
    # get time series of heating supply temperatures according to heating curve
    T_supply = grid.get_T_supply(param)

    
    for item in data["edges"]:
        d = data["edges"][item]["diameter_heating"]
        L = data["edges"][item]["length"]  
        t1 = pipes[d]["t_steel"]
        t2 = pipes[d]["t_ins"]
        t3 = pipes[d]["t_PE"]
                        
        if d == 0:
            k = 0
        else:   
            k = ( 1/param["conv_pipe"] +                                                                                                            # convection
                 d/2 * 1/param["lambda_steel"] * np.log((d+2*t1)/d) +                                                                               # thermal resistance steel pipe
                 d/2 * 1/param["lambda_ins"] * np.log((d+2*t1+2*t2)/(d+2*t1)) +                                                                     # thermal resistance insulation
                 d/2 * 1/param["lambda_PE"] * np.log((d+2*t1+2*t2+2*t3)/(d+2*t1+2*t2)) +                                                            # thermal resistance PE coat
                 d/2 * 1/param["lambda_soil"] * np.log((4*(param["grid_depth"] + param["R_0"]*param["lambda_soil"]))/(d+2*t1+2*t2+2*t3))) ** (-1)   # thermal restistance soil 
                
        losses["heating_grid"] = losses["heating_grid"] + k*np.pi*d*L*((T_supply - T_soil) + (param["T_heating_return"] - T_soil)) / 1e6


    #%% COOLING GRID LOSSES  
    
    losses["cooling_grid"] = np.zeros(8760)
    path = "input_data/pipes_cooling.txt"
    # available inner pipe diameters for the cooling network
    diameters = param["diameters"]["cooling"]
    # available pipe wall thicknesses for the cooling network
    t_PE = np.loadtxt(open(path, "rb"), delimiter = ",", usecols=(1))
    
    # create dictionary for cooling pipe geometry
    pipes = {}
    for i in range(np.size(diameters)):
        pipes[diameters[i]] = {"t_PE": t_PE[i]
                               }
    
    for item in data["edges"]:
        d = data["edges"][item]["diameter_cooling"]
        L = data["edges"][item]["length"]  
        t = pipes[d]["t_PE"]
            
        
        if d == 0:
            k = 0
        else:
            k = (1/param["conv_pipe"] +                                                                                                        # convection
                 d/2 * 1/param["lambda_PE"] * np.log((d+2*t)/d) +                                                                              # thermal resistance PE pipe
                 d/2 * 1/param["lambda_soil"] * np.log((4*(param["grid_depth"] + param["R_0"]*param["lambda_soil"]))/(d+2*t))) ** (-1)         # thermal resistance soil
                
        losses["cooling_grid"] = losses["cooling_grid"] + k*np.pi*d*L*((T_soil - param["T_cooling_supply"]) + (T_soil - param["T_cooling_return"])) / 1e6
        

    return losses
    


#%%
# Calculate and return time series of soil temperature in grid depth
# Model by M. Badache (2015): A new modeling approach for improved ground temperature profile determination
# http://dx.doi.org/10.1016/j.renene.2015.06.020
def calculateSoilTemperature(param): 
    
    
    # Load weather data
    path_weather = "input_data/weather.csv"
    weather = {}
    
    weather["T_air"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(0))          # Air temperatur °C
    weather["v_wind"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(1))         # Wind Velocity m/s
    weather["r"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(2))              # relative humidity -
    weather["G"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(3))              # Global radiation W/m^2
    weather["x"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(4))              # absolute humidity g/kg
    weather["p"] = np.loadtxt(open(path_weather, "rb"), delimiter = ",",skiprows = 1, usecols=(5))              # air pressure hPa
    
    
    # Calculate T_sky
    hours_day = np.arange(1,24)
    hours_day = np.append(hours_day,0)
    hours_year = []
    for i in range(365):
        hours_year = np.append(hours_year, hours_day)  #hours since midnight
    weather["p_w_dp"] = (weather["x"]/1000 * weather["p"]*100)/(0.622 + weather["x"]/1000)                                                                  # partial water pressure at dew point Pa
    weather["T_dp"] = (243.12*np.log(weather["p_w_dp"]/611.2))/(17.62-np.log(weather["p_w_dp"]/611.2))                                                      # dew point temperatur °C
    weather["T_sky"] = (weather["T_air"]+273.15)*((0.711+0.0056*weather["T_dp"]+0.000073*weather["T_dp"]**2+0.013*np.cos(15*hours_year))**0.25)-273.15      # sky temperature °C
    
    
    # Cosinus Fit of G, T_air and T_sky: X = mean - amp * cos(omega*t - phase)
    G_mean, G_amp, G_phase = cosFit(weather["G"])
    Tair_mean, Tair_amp, Tair_phase = cosFit(weather["T_air"])
    Tsky_mean, Tsky_amp, Tsky_phase = cosFit(weather["T_sky"])
    
#    print(G_mean)
#    print(G_amp)
#    print(G_phase)
#    print(Tair_mean)
#    print(Tair_amp)
#    print(Tair_phase)    
#    print(Tsky_mean)
#    print(Tsky_amp)
#    print(Tsky_phase) 
    
    
    # convective heat transfer at surface W/(m^2*K)
    weather["alpha_conv"] = np.zeros(8760)
    for hour in range(8760):
        if weather["v_wind"][hour] <= 4.88:
            weather["alpha_conv"][hour] = 5.7 + 3.8 * weather["v_wind"][hour]**0.5
        else:
            weather["alpha_conv"][hour] = 7.2*weather["v_wind"][hour]**0.78
    alpha_conv = np.mean(weather["alpha_conv"])
    
    # mean relative air humidity
    r = np.mean(weather["r"])
    
    
    # get ground parameters
    omega = 2*np.pi/365
    
    if param["asphaltlayer"] == 0:       # no asphalt layer, only soil
        alpha_s = param["alpha_soil"]
        epsilon_s = param["epsilon_soil"]
        f = param["evaprate_soil"]
        k = param["lambda_soil"]
        delta_s = (2*(k/param["heatcap_soil"]*3600*24)/omega)**0.5        # m,  surface damping depth
        delta_soil = delta_s
    else:                               # with asphalt layer at surface
        alpha_s = param["alpha_asph"]
        epsilon_s = param["epsilon_asph"]
        f = param["evaprate_asph"]
        k = param["lambda_asph"]
        delta_s = (2*(k/param["heatcap_asph"]*3600*24)/omega)**0.5                          # m,    surface damping depth (= asphalt)
        delta_soil = (2*(param["lambda_soil"]/param["heatcap_soil"]*3600*24)/omega)**0.5    # m,    soil damping depth       
    
    # simple correlation for average surface temperature 
    Ts_mean = (17.898 + 0.951 * (Tair_mean + 273.15)) - 273.15

    # long-wave radiation heat transfer
    alpha_rad = epsilon_s * 5.67e-8 * (Ts_mean + 273.15 + Tsky_mean + 273.15) * ((Ts_mean + 273.15)**2 + (Tsky_mean + 273.15)**2)
    
    h_e = alpha_conv*(1+103*0.0168*f) + alpha_rad
    h_r = alpha_conv*(1+103*0.0168*f*r)
    
    num = (h_r*Tair_amp+alpha_s*cmath.rect(G_amp,Tair_phase-G_phase)+alpha_rad*cmath.rect(Tsky_amp,Tair_phase-Tsky_phase))
    denom = (h_e+k*((1+1j)/delta_s))
    z = num/denom
    
    Ts_amp = abs(z)
    Ts_phase = Tair_phase + cmath.phase(z)       
 
    # Calculate soil temperature in grid depth
    d = param["d_asph"] 
    t = param["grid_depth"]
    omega = 2*np.pi/365/24
    time = np.arange(1,8761)
   
    if param["asphaltlayer"] == 0: # no asphalt
        weather["T_soil"] = Ts_mean - Ts_amp*np.exp(-t/delta_soil)*np.cos(omega*time - Ts_phase - t/delta_soil)
    else:   # with asphalt layaer
        if t > d: # grid is below asphalt layer
            weather["T_soil"] = Ts_mean - Ts_amp*np.exp(-d/delta_s)*np.exp(-(t-d)/delta_soil)*np.cos(omega*time - Ts_phase - d/delta_s - (t-d)/delta_soil)
        else: # grid is within asphalt layer
            weather["T_soil"] = Ts_mean - Ts_amp*np.exp(-t/delta_s)*np.cos(omega*time - Ts_phase - t/delta_s)
   
 
    #plt.plot(time/24, weather["T_soil"])
    #plt.show()

    
    T_soil = weather["T_soil"]
    
    print(T_soil)
    
    print(Ts_mean)
    print(Ts_amp)
    print(Ts_phase)
    
    
    return T_soil




#%%
def cosFit(data):
    
    omega = 2*np.pi/8760
    time = np.arange(1,8761)
    
    start_mean = np.mean(data)
    start_amp = np.std(data)* (2**0.5)
    start_phase = 0
    
    func = lambda x: x[0] - x[1]*np.cos(omega*time-x[2]) - data
    mean, amp, phase = leastsq(func, [start_mean, start_amp, start_phase])[0]
    
    #data_fit = mean - amp*np.cos(omega*time - phase)
    #plt.plot(time, data, '.')
    #plt.plot(time, data_fit)
    #plt.show()

    return mean, amp, phase


#%%








#%%