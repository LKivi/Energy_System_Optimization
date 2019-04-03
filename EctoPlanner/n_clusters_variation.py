# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 

@author: lkivi
"""

import parameters
import device_optim_ectogrid_clustered as opt
import datetime
import os
import numpy as np


use_case = "FZJ"


scenario = "Ectogrid_min"

# Choose scenario
#scenario = "typedays_1bal"                              # Typedays, single thermal balance for BU design, TES in buildings
#scenario = "typedays_absC"                              # Clustering, seperated heating and cooling balance for BU design, TES in buildings
#scenario = "typedays_1bal_noBldgTES"                    # Clustering, single thermal balance for BU design, no building storages
#scenario = "typedays_absC_noBldgTES"                    # Clustering, seperated heating and cooling balance for BU design, no building storages
#scenario = "typedays_standalone"
#scenario = "1bal_noBldgTES"



# Define paths
path_file = str(os.path.dirname(os.path.realpath(__file__)))
dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + "_" + scenario

if not os.path.exists(dir_results):
    os.makedirs(dir_results)
    
# Maximum number of typedays
N_max = 60
tac_opt = np.zeros(N_max)


for N in np.arange(1, N_max+1):

    ## Load parameters
    nodes, param, devs, devs_dom = parameters.load_params(use_case, path_file, scenario, N)
    
    # Run device optimization
    tac_opt[N-1] = opt.run_optim(nodes, param, devs, devs_dom, dir_results)