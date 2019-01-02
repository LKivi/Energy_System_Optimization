# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:23:57 2018

@author: mwi
"""

import numpy as np
import optim_model

 # Calculate mass flow through balancing unit (inter-balancing): "> 0": flow from supply to return pipe         
def design_balancing_unit(nodes, devs, param, residual, dir_results):
    
    time_steps = range(8760)
    dir_BU = dir_results + "\\Balancing_Unit"
    
    
    # Add thermal losses to residual loads
    residual["heat"] += param["heat_losses"]
    residual["cool"] += param["cool_losses"]
    
    # Set negative values to 0
    for t in time_steps:
        if residual["heat"][t] < 0:
            residual["heat"][t] = 0
        if residual["cool"][t] < 0:
            residual["cool"][t] = 0
    
    
    param, res_obj = optim_model.run_optim(devs, param, residual, dir_BU)
    
    print(res_obj)

    return nodes