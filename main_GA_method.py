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
import json
#from multiprocessing import Pool

time_start = time.clock()

"""
#==============================================================================
#                       Choose which Experiment to Run
#==============================================================================
"""

ExperimentNumberID = 1

"""
Step 0. Collect all boundary conditions and material constants for the experiment chosen
"""

import database_connection
BC = database_connection.Database.RetrieveBoundaryConditions(ExperimentNumberID)
MAT = database_connection.Database.RetrieveMaterial(BC.Material_ID[0])
CONSTRAINTS = database_connection.Database.RetrieveConstraints(BC.Constraint_ID[0])

"""
#==============================================================================
#                           Genetic algorithm START
#==============================================================================
"""

for Run in range(1,int(BC.NumberOfRuns)+1):     

    print("The algorithm has started run number "+str(Run))
    """
    Step 1. Initialize population
    """
    print("Step 1. Initializing initial population...")
    
    import genetic_algorithm
    Population = genetic_algorithm.Population(BC.N_pop[0]) #object initiated with its instance variables
    PopulationInitial = genetic_algorithm.Population.InitializePopulation(BC.NumberOfContainers[0], BC.Delta_a[0], BC.W[0], BC.N_pop[0], BC.T_dict[0], BC.SeedSettings[0], BC.SeedNumber[0], CONSTRAINTS) 
    
    for Generation in range(0,int(BC.NumberOfGenerations)): 
        print("Generation "+str(Generation)+" has started")
        """
        Step 2. Evaluate the fatigue fitness of the individuals in the population
        """
        print("Evaluating the objective function for each solution...")
        
        if Generation == 0:          #use Initial population for the first generation
        
            PopulationCurrent = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])
            
            for IndividualNumber in range(1,int(BC.N_pop)+1):
                PopulationCurrent.Fitness[IndividualNumber] = genetic_algorithm.GeneticAlgorithm.EvaluateFitnessFunction(BC.Fitness_Function_ID[0],PopulationInitial.Chromosome[IndividualNumber], BC.S_max[0], BC.a_0[0], BC.a_max[0], BC.Delta_a[0],MAT.C[0],MAT.m[0])
                PopulationCurrent.Chromosome[IndividualNumber] = PopulationInitial.Chromosome[IndividualNumber]
                
                # For comparison at the end of the optimisation, the fitness values for the initial population are also stored 
                PopulationInitial.Fitness[IndividualNumber] = PopulationCurrent.Fitness[IndividualNumber]
                
        else:                       #else use the Current population that has been produced through previous generations
        
            PopulationCurrent = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])
            
            for IndividualNumber in range(1,int(BC.N_pop)+1):
                
                PopulationCurrent.Fitness[IndividualNumber] = genetic_algorithm.GeneticAlgorithm.EvaluateFitnessFunction(BC.Fitness_Function_ID[0],PopulationOffspring.Chromosome[IndividualNumber], BC.S_max[0], BC.a_0[0], BC.a_max[0], BC.Delta_a[0],MAT.C[0],MAT.m[0])
                PopulationCurrent.Chromosome[IndividualNumber] = PopulationOffspring.Chromosome[IndividualNumber]
        
                
        """
        Step 2.a Checking the Termination Condition 
        """
              
        TerminationCondition = genetic_algorithm.GeneticAlgorithm.CheckTermination(PopulationCurrent, Generation)
        
        if TerminationCondition == True:
            
            PopulationFinal = PopulationCurrent
        
        else: 
            # Store information on the current population for the family tree
            pass
        """
        Step 3. Select the fittest solutions
        """
        PopulationCurrentSelected = genetic_algorithm.GeneticAlgorithm.SelectSurvivingPopulation(PopulationCurrent, BC.Rs[0])
    
        if BC.Crossover[0] == str(True):
        
            """
            Step 4. Determine probability of reproduction for each solution
            """
            
            PopulationParents = genetic_algorithm.GeneticAlgorithm.CalculateReproductionProbParents(PopulationCurrentSelected, BC.RankingOperator[0])
            
            """
            Step 5. Select (two) solutions from parent population for reproduction
            """
            
            PopulationOffspring = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])
    
            while PopulationOffspring.tail(1).Chromosome[BC.N_pop[0]] is None: #keep recombining until the entire PopulationOffspring is filled with solutions
                
                ParentSelected1, ParentSelected2 = genetic_algorithm.GeneticAlgorithm.SelectParents(PopulationParents)
                
                """
                Step 6. Crossover of the selected parent solutions
                """
                
                PopulationOffspring = genetic_algorithm.GeneticAlgorithm.RecombineParents(PopulationParents.Chromosome[ParentSelected1], PopulationParents.Chromosome[ParentSelected2], PopulationOffspring, BC.Pc[0], BC.W[0], BC.CrossoverOperator[0], CONSTRAINTS, BC.NumberOfContainers[0])
            
        else:
            PopulationOffspring = PopulationCurrentSelected # if crossover has been disabled, the surviving population becomes the offspring population
            
        """
        Step 7. Mutation of Offspring population
        """
        
        if BC.Mutation[0] == str(True):
        
            for IndividualNumber in range(1,len(PopulationOffspring)+1):
                print("Starting mutation of the Offspring Population for Individual...", IndividualNumber)
                PopulationOffspring.Chromosome[IndividualNumber] = genetic_algorithm.GeneticAlgorithm.MutatePopulation(PopulationOffspring.Chromosome[IndividualNumber], BC.MutationOperator[0], BC.Pm[0], BC.NumberOfContainers[0], BC.W[0],  BC.T_dict[0], CONSTRAINTS)
        

    """
    #==============================================================================
    #                              Genetic Algorithm END
    #==============================================================================
    """    
    
"""
#==============================================================================
#                           Visualization of Results
#==============================================================================
"""  

import visuals

# Sort the Final Population based on Fitness

PopulationFinal = PopulationCurrent.sort_values("Fitness", ascending= False, kind='mergesort')

# Show top 3 crenellation patterns in the Final Population

visuals.FatigueVisuals.ShowTop3CrenellationPatterns(PopulationFinal, PopulationInitial)


"""
Store the GA data per run to CSV file
"""

convergence_overview.to_pickle("reference_Lu_GA_convergence_run_"+str(run)+"")


"""  
#==============================================================================
#                          Runtime of the algorithm
#==============================================================================
"""
time_elapsed = (time.clock() - time_start)
print("Calculation time:", round(time_elapsed,2), "seconds, or ",round(time_elapsed/60,2)," minutes")


