# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 02:28:07 2019

@author: lkivi
"""


import numpy as np
import os
import parameters
import datetime

path_file = str(os.path.dirname(os.path.realpath(__file__)))
dir_results = path_file + "\\Results\\" + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

if not os.path.exists(dir_results):
    os.makedirs(dir_results)


# Choose use case
use_case = "FZJ"


# Load parameters
nodes, param = parameters.load_params(use_case, path_file)




typedays = []



z = param["day_matrix"]


for d in range(365):
    if any(z[d]):
        typedays.append(d)
        

#matches = np.zeros((param["n_clusters"], 2))
#matches[:,0] = np.arange(param["n_clusters"])
#matches[:,1] = typedays

dem_heat_clustered = np.sum(nodes[n]["heat"] for n in nodes)

dem_heat = np.zeros(8760)

for d in range(365):
    match = np.where(z[:,d] == 1)[0][0]
    typeday = np.where(typedays == match)[0][0]
    
    dem_heat[24*d:24*(d+1)] = dem_heat_clustered[typeday,:]
    
    