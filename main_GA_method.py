#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:46:57 2017

@author: Bart van der Lee
@project: MSc thesis 

#==============================================================================
#                          Import packages 
#==============================================================================
"""

import numpy as np
import time
from numpy import *
import matplotlib.pyplot as pp
import pandas as pd
#from multiprocessing import Pool

time_start = time.clock()

"""
#==============================================================================
#                       Choose which Experiment to Run
#==============================================================================
"""

ExperimentNumberID = 1

"""
Step 0. Collect all boundary conditions for the experiment chosen
"""

import database
BC = database.Database.RetrieveBoundaryConditions(ExperimentNumberID)

"""
#==============================================================================
#                           Genetic algorithm START
#==============================================================================
"""

for Run in range(1,BC.NumberOfRuns+1):     

    print("The algorithm has started run number "+str(Run))
    """
    Step 1. Initialize population
    """
    print("Step 1. Initializing initial population...")
    
    import genetic_algorithm
    Population = genetic_algorithm.Population(BC.N_pop, BC.PopStatisticsDict) #object initiated with its instance variables
    PopulationInitial = genetic_algorithm.Population.InitializePopulation(BC.delta_x, BC.W, BC.N_pop, BC.t_dict, BC.SeedSettings, BC.SeedNumber) 
        
    for Generation in range(0,BC.NumberOfGenerations): 
        print("Generation "+str(Generation)+" has started")
        """
        Step 2. Evaluate the fatigue fitness of the individuals in the population
        """
        print("Evaluating the objective function for each solution...")
        
        import fatigue
            
        if Generation == 0:          #use Initial population for the first generation
        
            PopulationCurrent = database.Database.RetrievePopulationDataframe()
            
            for IndividualNumber in range(1,BC.N_pop+1):
            
                PopulationCurrent.Fitness[IndividualNumber] = fatigue.Fatigue.CalculateFatigueLife(PopulationInitial.Chromosome[IndividualNumber], BC.S_max, BC.a_0, BC.a_max, BC.delta_a,BC.C,BC.m)
            
        else:                       #else use the Current population that has been produced through previous generations
        
            PreviousPopulation = PopulationCurrent
            PopulationCurrent = database.Database.RetrievePopulationDataframe()
            
            for Individual in range(1,BC.N_pop+1):
                
                PopulationCurrent = fatigue.Fatigue.CalculateFatigueLife(PopulationCurrent)
       
        """
        Step 2.a Store evaluated individuals for visualizations
        """  
    #    population.generation_data(population_parents_evaluated, g, NumberOfGenerations)
        
        """
        Step 3. Select the fittest solutions
        """
        PopulationCurrentSelected = genetic_algorithm.GeneticAlgorithm.SelectSurvivingPopulation(PopulationCurrent, BC.Rs)
    
        """
        Step 4. Determine probability of reproduction for each solution
        """
        
        PopulationParents = genetic_algorithm.CalculateSelectionProbParents(PopulationCurrentSelected)
        
        
        """
        Step 5. Select solutions from parent population for reproduction
        """
        
        PopulationOffspring = database.Database.RetrievePopulationDataframe()

        while len(PopulationOffspring) < len(PopulationParents):
            
            ParentsSelected = genetic_algorithm.SelectParents(PopulationParents)
            
            """
            Step 6. Crossover of the selected parent solutions
            """
            
            PopulationOffspring = genetic_algorithm.GeneticAlgorithm.RecombineParents(ParentsSelected, PopulationOffspring, BC.Pc)
            
            
        """
        Step 7. Mutation of Offspring population
        """
        
        PopulationOffspringMutated = genetic_algorithm.GeneticAlgorithm.MutatePopulation(PopulationOffspring, BC.Pm)
        
    
        
        """
        Step 7.a Checking the Termination Condition for a Run
        """
        
        TerminationCondition = genetic_algorithm.GeneticAlgorithm.CheckTermination()
        
        if TerminationCondition == True:
            
            PopulationFinal = PopulationOffspringMutated
            
        
            #insert loop to evaluate fitness function of each individual
            
            break
        
        else: #store Offspring Population as the Previous Population for the next Generation
            
            #insert line to store the population information in the database
        
            PopulationPrevious = PopulationOffspringMutated
            
            continue
            
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


