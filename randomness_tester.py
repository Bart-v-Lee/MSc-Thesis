#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:52:32 2018

@author: Bart
"""
import numpy as np

np.random.seed(45)

NumberArrayUniform = []
NumberArrayChoice = []


while len(NumberArrayUniform) is not 20:
    NumberArrayUniform.append(np.random.uniform(0,1))
    
    
    # random uniform
    
    NumberArrayChoice.append(np.random.choice([1,2,3,4,5,6,7,8,9,10]))
    
    
print(NumberArrayUniform)
print(NumberArrayChoice)
    