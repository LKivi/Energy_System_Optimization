# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 17:31:57 2018

@author: mwi
"""

import parameters
import bldg_balancing_optim_complete as opt_compl
import os
import datetime
import json
import numpy as np

path_file = str(os.path.dirname(os.path.realpath(__file__)))
dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

if not os.path.exists(dir_results):
    os.makedirs(dir_results)


# Choose use case
use_case = "FZJ"

# Load parameters
nodes, param, devs, devs_dom, time_steps = parameters.load_params(use_case, path_file)

opt_compl.run(nodes, param, devs, devs_dom, dir_results)


sum_residual_heat = np.zeros(8760)
sum_power_dem_bldgs = np.zeros(8760)
for t in time_steps:
    sum_residual_heat[t] = sum(nodes[n]["res_heat_dem"][t] for n in range(len(nodes)))
    sum_power_dem_bldgs[t] = sum(nodes[n]["power_dem"][t] for n in range(len(nodes))) 


# Network residual loads
residual = {}
residual["heat"] = np.zeros(8760)
residual["cool"] = np.zeros(8760)
for t in time_steps:
    if sum_residual_heat[t] > 0:
        residual["heat"][t] = sum_residual_heat[t] / 1000           # MW, total residual heat demand
    else:
        residual["cool"][t] = (-1) * sum_residual_heat[t] / 1000    # MW, total residual cooling demand
residual["power"] = sum_power_dem_bldgs / 1000                      # MW, total electricity demand for devices in buildings


#with open(dir_results + "\\residuals.json", "w") as outfile:
#        json.dump(residual, outfile, indent=4, sort_keys=True)
