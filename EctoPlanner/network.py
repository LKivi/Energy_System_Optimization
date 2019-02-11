# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:35:09 2019

@author: lkivi
"""

import network_optim
import network_optim_clustered


def design_network(nodes, param, dir_results):
    
    if param["switch_clustered"]:
        param = network_optim_clustered.run_optim(nodes, param, dir_results)
    else:
        param = network_optim.run_optim(nodes, param, dir_results)
        
        