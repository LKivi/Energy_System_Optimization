# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:13:43 2018

@author: mwi
"""


import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

im1 = np.memmap("1_Lageplan.PNG", dtype=np.uint8, shape=(768, 1024, 3))



im_flow = np.memmap("Mass_flow_t0.png", dtype=np.uint8, shape=(218, 439, 3))

plt.imshow(im_flow)#, cmap=plt.cm.gray)  
