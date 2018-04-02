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
#check the hard-coded EXPERIMENT ID in StorePopulationComposition method!
#check the hard-coded EXPERIMENT ID in ShowTop3CrenellationPatterns method!

ExperimentNumberID = 1
np.random.seed(45)


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
        
#    Population = genetic_algorithm.Population(BC.N_pop[0]) #object initiated with its instance variables
    PopulationInitial = genetic_algorithm.Population.InitializePopulation(BC.n_total[0], BC.Delta_a[0], BC.W[0], BC.N_pop[0], BC.T_dict[0], BC.SeedSettings[0], BC.SeedNumber[0], CONSTRAINTS) 
    
    # Initialize the PopulationComposition dictionary
    PopulationComposition = genetic_algorithm.Population.CreatePopulationCompositionDictionary(BC.NumberOfGenerations[0])
        
    for Generation in range(0,int(BC.NumberOfGenerations[0])): 
        
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

                # Store information into the PopulationComposition dictionary for current individual
                PopulationComposition['Gen '+str(0)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(0)][0], Operation = "Initialization", Chromosome = PopulationInitial.Chromosome[IndividualNumber] , Fitness = PopulationInitial.Fitness[IndividualNumber], Pp = None, Parents = 0)
                
#                for UniqueChromosome in range(0,len(PopulationComposition['Gen 0'][0])):
#                    
#                    PopulationComposition['Gen 0'][0].loc[UniqueChromosome,"Genotype"] = genetic_algorithm.Population.ExtractGenotype(PopulationComposition['Gen 0'][0].loc[UniqueChromosome,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
#         
        else:                       #else use the Current population that has been produced through previous generations
        
            PopulationCurrent = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])
            
            for IndividualNumber in range(1,int(BC.N_pop)+1):
                
                PopulationCurrent.Fitness[IndividualNumber] = genetic_algorithm.GeneticAlgorithm.EvaluateFitnessFunction(BC.Fitness_Function_ID[0],PopulationOffspring.Chromosome[IndividualNumber], BC.S_max[0], BC.a_0[0], BC.a_max[0], BC.Delta_a[0],MAT.C[0],MAT.m[0])
                
                PopulationCurrent.Chromosome[IndividualNumber] = PopulationOffspring.Chromosome[IndividualNumber]
        
                #Store information into the PopulationComposition dictionary
                PopulationComposition['Gen '+str(Generation)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation)][0], Operation = "Evaluation", Chromosome = PopulationCurrent.Chromosome[IndividualNumber] , Fitness = PopulationCurrent.Fitness[IndividualNumber], Pp = None, Parents = 0)
  
                
        """
        Step 2.a Checking the Termination Condition 
        """
              
        TerminationCondition = genetic_algorithm.GeneticAlgorithm.CheckTermination(BC.NumberOfGenerations[0], Generation, PopulationCurrent)
        
        if TerminationCondition == True:
            
            #PopulationComposition['Gen '+str(Generation)][1] = genetic_algorithm.Population.StoreGeneComposition(PopulationComposition['Gen '+str(Generation)][0], BC.T_dict[0], BC.n_total[0])
            
            """
            Extract the genotypes of all final chromosomes and add them to the PopulationComposition dictionary for analysis. 
            Calculate PopulationGeneComposition array for analysis.
            """
            
            for UniqueChromosome in range(0,len(PopulationComposition['Gen '+str(Generation)][0])):
#                print("Genotypes have been extracted...")
#                print("Generation number ",Generation)
#                print(UniqueChromosome, "unique chromosome")
#                print(PopulationComposition['Gen '+str(Generation)][0])
#                print(PopulationComposition['Gen '+str(Generation)][0].loc[UniqueChromosome,"Chromosome"])
                
                PopulationComposition['Gen '+str(Generation)][0].Genotype[UniqueChromosome] = genetic_algorithm.Population.ExtractGenotype(PopulationComposition['Gen '+str(Generation)][0].loc[UniqueChromosome,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
                    
            # Calculate and Store GeneComposition for the current generation  
            
            PopulationComposition['Gen '+str(Generation)][1] = genetic_algorithm.Population.StoreGeneComposition(PopulationComposition['Gen '+str(Generation)][0], BC.T_dict[0], BC.n_total[0])
          

            PopulationFinal = PopulationCurrent
            
            break

        
        else: 
            # Store information on the current population for the family tree
            
            """
            Transfer memory of unique chromosomes, fitness and genotypes to the next generation
            """
            
            PopulationComposition = genetic_algorithm.Population.TransferPopulation(Generation, PopulationComposition)
            
            
            """
            Step 3. Select the fittest solutions
            """
            PopulationCurrentSelected = genetic_algorithm.GeneticAlgorithm.SelectSurvivingPopulation(PopulationCurrent, BC.Rs[0])
        
            if BC.Crossover[0] == str(True):
            
                """
                Step 4. Determine probability of reproduction for each solution
                """
                
                PopulationParents = genetic_algorithm.GeneticAlgorithm.CalculateReproductionProbParents(PopulationCurrentSelected, BC.RankingOperator[0])
                
                # Store population data Pp to PopulationComposition dataframe
                
                for IndividualNumber in range(1,len(PopulationParents)+1):
                
                    PopulationComposition['Gen '+str(Generation)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation)][0], Operation = "Selection", Chromosome = PopulationParents.loc[IndividualNumber,"Chromosome"] , Fitness = None, Pp = PopulationParents.loc[IndividualNumber, "Pp"], Parents = None)
    
                
                """
                Step 5. Select (two) solutions from parent population for reproduction
                """
                
                PopulationOffspring = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])
                
#                print(PopulationOffspring, "empty Population Offspring")
                
                NumberOfRecombinations = []
        
                while PopulationOffspring.tail(1).Chromosome[BC.N_pop[0]] is None: #keep recombining until the entire PopulationOffspring is filled with solutions
                    
                    ParentSelected1, ParentSelected2 = genetic_algorithm.GeneticAlgorithm.SelectParents(PopulationParents)
                    

                    """
                    Step 6. Crossover of the selected parent solutions
                    """
                    
                    PopulationOffspring, Child1, Child2 = genetic_algorithm.GeneticAlgorithm.RecombineParents(PopulationParents.Chromosome[ParentSelected1], PopulationParents.Chromosome[ParentSelected2], PopulationOffspring, BC.Pc[0], BC.W[0], BC.CrossoverOperator[0], CONSTRAINTS, BC.n_total[0])
                
                    # Store the offspring relations in the PopulationComposition dictionary

                    Children = [Child1, Child2]
                    for Child in range(0,len(Children)):
                        PopulationComposition['Gen '+str(Generation+1)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation+1)][0], Operation = "Crossover", Chromosome = Children[Child] , Fitness = 0, Pp = None, Parents = [PopulationParents.Chromosome[ParentSelected1],PopulationParents.Chromosome[ParentSelected2]])
    
                            
            # if Crossover is not enabled, the surviving population becomes the offspring population. Bart: The size still needs to increase with a size, depending on Rs!
                            
            else:
                PopulationOffspring = PopulationCurrentSelected # if crossover has been disabled, the surviving population becomes the offspring population
                    
                # Store the offspring relations into the PopulationComposition dictionary
                
                for IndividualNumber in range(1,len(PopulationOffspring)):
                    
                    PopulationComposition['Gen '+str(Generation+1)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation+1)][0], Operation = "Crossover", Chromosome = Children[Child] , Fitness = 0, Pp = None, Parents = [PopulationParents.Chromosome[ParentSelected1],PopulationParents.Chromosome[ParentSelected2]])
                    
                
            """
            Step 7. Mutation of Offspring population
            """
            
            if BC.Mutation[0] == str(True):
            
                for IndividualNumber in range(1,len(PopulationOffspring)+1):
                    print("Starting mutation of the Offspring Population for Individual...", IndividualNumber)
                    ParentChromosome = PopulationOffspring.Chromosome[IndividualNumber]

                    ChildChromosome = genetic_algorithm.GeneticAlgorithm.MutatePopulation(PopulationOffspring.Chromosome[IndividualNumber], BC.MutationOperator[0], BC.Pm[0], BC.n_total[0], BC.W[0],  BC.T_dict[0], CONSTRAINTS)
                    PopulationOffspring.Chromosome[IndividualNumber] = ChildChromosome
                    
                    Identical = None
                    if np.array_equal(ParentChromosome.Thickness,ChildChromosome.Thickness):
                        Identical = True
                    else:
                        Identical = False
                    # Store the relations after mutating into the PopulationComposition dictionary

                    PopulationComposition['Gen '+str(Generation+1)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation+1)][0], Operation = "Mutation", Chromosome = PopulationOffspring.Chromosome[IndividualNumber] , Fitness = 0, Pp = None, Parents = [ParentChromosome])

    
            """
            Extract the genotypes of all new chromosomes and add them to the PopulationComposition dictionary for analysis. 
            Calculate PopulationGeneComposition array for analysis.
            """
            
            for UniqueChromosome in range(0,len(PopulationComposition['Gen '+str(Generation)][0])):
#                print("Genotypes have been extracted...")
#                print("Generation number ",Generation)
#                print(UniqueChromosome, "unique chromosome")
#                print(PopulationComposition['Gen '+str(Generation)][0])
#                print(PopulationComposition['Gen '+str(Generation)][0].loc[UniqueChromosome,"Chromosome"])
                
                PopulationComposition['Gen '+str(Generation)][0].Genotype[UniqueChromosome] = genetic_algorithm.Population.ExtractGenotype(PopulationComposition['Gen '+str(Generation)][0].loc[UniqueChromosome,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
                    
            # Calculate and Store GeneComposition for the current generation  
            
            PopulationComposition['Gen '+str(Generation)][1] = genetic_algorithm.Population.StoreGeneComposition(PopulationComposition['Gen '+str(Generation)][0], BC.T_dict[0], BC.n_total[0])
          
            

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

#convergence_overview.to_pickle("reference_Lu_GA_convergence_run_"+str(run)+"")


"""  
#==============================================================================
#                          Runtime of the algorithm
#==============================================================================
"""
time_elapsed = (time.clock() - time_start)
print("Calculation time:", round(time_elapsed,2), "seconds, or ",round(time_elapsed/60,2)," minutes")


