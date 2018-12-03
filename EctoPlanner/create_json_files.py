# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:37:01 2018

@author: mwi
"""

import json
import time as time

use_case = "simple_example"

time_stamp = time.ctime

#%% Use case: EON (example data provided by E.ON)
if use_case == "simple_example":
    
    meta = {"description": "Simple example with only a few nodes",
            "name": "simple_example",
            "source": "mwi_manual", 
            "created": time_stamp, # to string!
            "units": {"diameter": "m", "length":"m"},
            }

    nodes = {}
    
    nodes[0] = {"lon":      1, # longitude
                "lat":      0, # latitude 
                "name":  "test_bldg_0",
                "commodities": ["heat", "cool", "power"],
                }
    
    nodes[1] = {"lon":      1, # longitude
                "lat":      1, # latitude 
                "name":  "test_bldg_1",
                "commodities": ["heat", "cool", "power"],
                }
    
    nodes[2] = {"lon":      0, # longitude
                "lat":      0, # latitude 
                "name":  "test_bldg_2",
                "commodities": ["heat", "cool", "power"],
                }
    
    nodes[3] = {"lon":      0, # longitude
                "lat":      1, # latitude 
                "bldg_id":  "test_bldg_3",
                "commodities": ["heat", "cool", "power"],
                }
    
    edges[] = 
    
    print("Using test data set.")
    nodes = {}

nodes = json.loads(open("D:\\mwi\\Gurobi_Modelle\\EctoPlanner\\input_data\\nodes.json").read())


nodes2 = json.loads(open("D:\\mwi\\Gurobi_Modelle\\EctoPlanner\\input_data\\nodes_2.json").read())