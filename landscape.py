#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:00:33 2018

@author: bart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from mpl_toolkits import mplot3d

import fatigue
import visuals


fig = pp.figure()
ax = pp.axes(projection='3d')

n1 = [0,1,2]
n2 = [0,1,2]
n3 = [0,1,2]
N3 = 1
    

N1, N2 = np.meshgrid(n1,n2)

Fitness = visuals.LandscapeVisuals.ReturnLandscapeValue(N1,N2,N3)

ax.contour3D(N1, N2, Fitness, 50, cmap='binary')

ax.set_xlabel('n_1 thickness level')
ax.set_ylabel('n_2 thickness level')
ax.set_zlabel('Fitness')