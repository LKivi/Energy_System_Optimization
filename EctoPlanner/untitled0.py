# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 23:38:55 2019

@author: lkivi
"""

import numpy as np

cool = np.zeros(8760)
heat = np.zeros(8760)

for n in nodes:
    for t in time_steps:
        if nodes[n]["res_heat_dem"][t] >= 0:
            heat[t] += nodes[n]["res_heat_dem"][t]
        else:
            cool[t] += (- nodes[n]["res_heat_dem"][t])
            

cool = cool / 1000
heat = heat / 1000