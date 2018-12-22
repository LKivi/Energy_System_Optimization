# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 17:31:57 2018

@author: mwi
"""

import parameters
import global_optimization
import os
import datetime
import json

path_file = str(os.path.dirname(os.path.realpath(__file__)))
dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

if not os.path.exists(dir_results):
    os.makedirs(dir_results)


# Choose use case
#use_case = "FZJ"

# Load parameters
#nodes, param, devs, devs_dom, time_steps = parameters.load_params(use_case, path_file)

#global_optimization.run_optim(nodes, param, devs, devs_dom, dir_results)



with open(dir_results + "\residuals.json", "w") as outfile:
        json.dump(residual, outfile, indent=4, sort_keys=True)