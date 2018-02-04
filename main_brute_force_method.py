#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:11:49 2017

@author: Bart
"""

import numpy as np
import time
from numpy import *
import matplotlib.pyplot as pp
import sqlite3
import pandas as pd
#from multiprocessing import Pool

time_start = time.clock()

from visuals import visual
from crenellation import crenellation
from fatigue import fatigue
from genetic_algorithm import genetic_algorithm
from brute_force_method import brute_force_method

#==============================================================================
# Extract boundary conditions from SQL database
#==============================================================================

conn = sqlite3.connect("materials.db")
cur =  conn.cursor()
m1 = pd.read_sql_query("Select * from Materials;", conn)
bc = pd.read_sql_query("Select * from BD;", conn)
rd = pd.read_sql_query("Select * from Reference_data;", conn)
m2 = m1.set_index("Source")
m2 = m2.loc['efatigue']
bc = bc.set_index("Type")
bc = bc.loc['Reference Study Lu (2015) - Crenellation Brute force']
conn.close

#==============================================================================
# Calculation of all possible crenellation patterns
#==============================================================================

brute_force = brute_force_method(bc)
"""
Simple
"""
#population = brute_force.brute_force_method_simple(bc)
"""
Refined reference
"""
population = brute_force.brute_force_method_reference_5cont_8thick_stringers(bc)

"""
Fatigue evaluation of entire population
"""

fatigue = fatigue(population, bc, m2)
population_eval = fatigue.fatigue(population,bc,m2)


"""
Sort based on fitness level
"""

population_eval_sorted = population_eval.sort_values("Fitness",ascending = False, kind = 'mergesort')
population_eval_sorted.Fitness = population_eval_sorted.Fitness.round(decimals =2)

"""
Save population brute force to csv or pickle it
"""
population_eval_sorted.to_csv("reference_Lu_refined_brute_force")

population_eval_sorted.to_pickle("reference_Lu_brute_force_5cont_8thick_stringers")


"""
Read historic data back to Spyder
"""

population_eval_sorted_5cont_8thick = pd.read_pickle("/Users/Bart/Google Drive/6. Education/Master/4. MSc Thesis/Python code/classes/version2/reference_Lu_brute_force_5cont_8thick")
