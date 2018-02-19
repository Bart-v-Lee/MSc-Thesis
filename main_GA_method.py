#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:46:57 2017

@author: Bart van der Lee
@project: MSc thesis 

#==============================================================================
#                          Import packages and classes
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
from crenellation import CrenellationPattern
from fatigue import fatigue
from genetic_algorithm import genetic_algorithm
from database import database

"""
#==============================================================================
#                       Choose which Experiment to Run
#==============================================================================
"""

# ExperimentNumberID = 

"""
Step 0. Collect all boundary conditions for the experiment chosen
"""

delta_x, W, N_pop, t_dict, SeedSettings, SeedNumber, NumberOfRuns, NumberOfGenerations = Database.RetrieveBoundaryConditions(ExperimentNumberID)

"""
#==============================================================================
#                           Genetic algorithm START
#==============================================================================
"""

for Run in range(1,NumberOfRuns+1): #number of times that the genetic algorithm is run for the same experiment

    print("The algorithm has started run number "+str(Run))
    """
    Step 1. Initialize population
    """
    print("Step 1. Initializing initial population...")
    
    Crenellation = CrenellationPattern() #object initiated with its given attributes
    PopulationInitial = Crenellation.InitializePopulation(delta_x, W, N_pop, t_dict, SeedSettings, SeedNumber) 
        
    
    for Generation in range(0,NumberOfGenerations): 
        print("Generation "+str(Generation)+" has started")
        """
        Step 2. Evaluate the fatigue fitness of the individuals in the population
        """
        print("Evaluating the objective function for each solution...")
        
        Fatigue = FatigueCalculations(bc,material,population) 
            
        if g ==0:   #use initial population for the first generation
        
            #insert loop for going through each individual in the population
            PopulationCurrent = Fatigue.CalculateFatigueLife(PopulationInitial, bc, material)
            
        else:       #else use the current population that has been produced through previous generations
        
            #insert loop for going through each individual in the population
            PopulationCurrent = Fatigue.CalculateFatigueLife(PopulationCurrent, bc,material)
       
        """
        Step 2.a Store evaluated individuals for visualizations
        """  
    #    population.generation_data(population_parents_evaluated, g, NumberOfGenerations)
        
        """
        Step 3. Select the fittest solutions
        """
        #PopulationCurrentSelected = 
    
        """
        Step 4. Determine probability of reproduction for each solution
        """
        
        PopulationParents = population.CalculateSelectionProbParents(bc, population_parents_evaluated)
        
        """
        Step 5. Select solutions from parent population for reproduction
        """
        
        PopulationParentsSelected = population.recombination(bc,material,population_parents_ranked)
        
        """
        Step 6. Crossover of the selected parent solutions
        """
        
        #PopulationOffspring = 
        
        """
        Step 6.a Checking the Recombination Condition for a Generation
        """
        
        
        
        """
        Step 7. Mutation of Offspring population
        """
        
        #PopulationOffspringMutated = 
        
        """
        Step 7.a Checking the Termination Condition for a Run
        """
        
        #PopulationFinal = 
        
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


