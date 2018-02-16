#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:46:57 2017

@author: Bart van der Lee

#==============================================================================
# Import packages and classes
#==============================================================================
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

"""
#==============================================================================
# Import boundary conditions from SQL database
#==============================================================================
"""

conn = sqlite3.connect("materials.db")
cur =  conn.cursor()
material = pd.read_sql_query("Select * from Materials;", conn) 
bc = pd.read_sql_query("Select * from BD;", conn)
#rd = pd.read_sql_query("Select * from Reference_data;", conn)

material = material.set_index("Source")
material = material.loc['efatigue']

bc = bc.set_index("Type")
bc = bc.loc['Reference Study Lu (2015) - Crenellation GA']

conn.close

"""
#==============================================================================
# Genetic algorithm START
#==============================================================================
"""

number_of_runs = int(bc.ix["number_of_runs"])
population_children = [] #initialize empty array for children population
convergence_overview = [] #initialize empty array for convergence tests

for run in range(1,number_of_runs+1): #number of times that the genetic algorithm is run seperately
   
    """
    Step 1. Initialize population
    """
    Crenellation = CrenellationPattern(bc, material)
    PopulationInitial = Crenellation.InitializePopulation(bc,material,population) 
    
    NumberOfGenerations = int(bc.ix["Number of Generations"])
    
    for g in range(0,NumberOfGenerations): 
        print("Generation "+str(g)+" has started")
        """
        Step 2. Evaluate the fatigue fitness of the individuals in the population
        """
        Fatigue = FatigueCalculations(bc,material,population) 
            
        if g ==0:   #use initial population if generation number is zero
            #insert loop for going through each individual in the population
            PopulationParents = Fatigue.CalculateFatigueLife(PopulationInitial, bc, material)
            
        else:       #else use children population from previous generation
        
            #insert loop for going through each individual in the population
            PopulationParents = Fatigue.CalculateFatigueLife(PopulationCurrent, bc,material)
       
        """
        Step 2.a Store evaluated individuals for visualizations
        """  
    #    population.generation_data(population_parents_evaluated, g, NumberOfGenerations)
        
        """
        Step 3. Select the fittest solutions
        """
    
    
        """
        Step 4. Determine probability of reproduction for each solution
        """
        
        population_parents_ranked = population.rank_parents(bc, population_parents_evaluated)
        
        """
        Step 5. Select solutions from parent population for reproduction
        """
        
        population_children = population.recombination(bc,material,population_parents_ranked)
        
        """
        Step 6. Crossover of the selected parent solutions
        """
        
        #select half of the population for survival
        
        """
        Step 7. Mutation of Offspring population
        """
        
        """
        Step 7.a Checking the Termination Condition for a Run
        """
        
        """
        Step 7.b Checking the Termination Condition for the Algorithm
        """
        
        termination_overview = population.population_convergence(bc, population_parents_evaluated, g, convergence_overview)

    """
    #==============================================================================
    # Genetic Algorithm END
    #==============================================================================
    """    
        
    """
    #==============================================================================
    # Visualization of Results
    #==============================================================================
    """  
    
    visuals = visual(population_parents_evaluated, bc,material, population, run , g)

    #fitness_plot = visual.fitness_plot(population_eval,g)
    crennellation_fittest = visual.fittest_crenellation(convergence_overview, run)

    #individual_plot = visual.create_plot(population_eval, bc,material, individual_no=8)
    #individual_plot = visual.create_plot(population_eval, bc,material, individual_no=4)
    
    convergence_overview_plot = visuals.convergence(convergence_overview, run, bc)
    
    """
    Store the GA data per run to CSV file
    """
    #convergence_overview.to_csv("reference_Lu_GA_convergence_run_"+str(run)+"")
    convergence_overview.to_pickle("reference_Lu_GA_convergence_run_"+str(run)+"")


"""  
#==============================================================================
# Runtime of the algorithm
#==============================================================================
"""
time_elapsed = (time.clock() - time_start)
print("Calculation time:", round(time_elapsed,2), "seconds, or ",round(time_elapsed/60,2)," minutes")


