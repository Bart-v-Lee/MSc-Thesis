#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:46:57 2017

@author: Bart van der Lee
@project: MSc thesis 
"""

"""
#==============================================================================
#                          Import packages 
#==============================================================================
"""

import numpy as np
import pandas as pd
import time
#import matplotlib.pyplot as pp
#from multiprocessing import Pool

time_start = time.clock()

class MainProgramme:

    def RunGeneticAlgorithm(ExperimentNumberID, BC, MAT, CONSTRAINTS):
          
        """
        #==============================================================================
        #                           Genetic algorithm START
        #==============================================================================
        """
        

        """
        Step 1. Initialize population
        """
        print("Step 1. Initializing initial population...")
        
        import database_connection
        import genetic_algorithm
            
        PopulationInitial = genetic_algorithm.Population.InitializePopulation(BC.n_total[0], BC.Delta_a[0], BC.W[0], BC.N_pop[0], BC.T_dict[0], BC.SeedSettings[0], BC.SeedNumber[0], CONSTRAINTS, ExperimentNumberID) 
        
        PopulationOffspring = None #start out with an empty PopulationOffspring
                    
        # Initialize the PopulationComposition dictionary
        
        PopulationComposition = genetic_algorithm.Population.CreatePopulationCompositionDictionary(BC.NumberOfGenerations[0])
        
        # Initialize fitness function evaluation counter
        
        NumberOfEvaluations = 0
            
        for Generation in range(0,int(BC.NumberOfGenerations[0])): 
            
            print("Generation "+str(Generation)+" has started")
            
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            
            Step 2. Evaluation -  assign a fitness value to each solution in the population
            a
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                            
            if Generation == 0:          # use Initial population for the first generation
            
                PopulationCurrent = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])
                
                for IndividualNumber in range(1,int(BC.N_pop)+1):
                    PopulationCurrent.Fitness[IndividualNumber], FatigueCalculations = genetic_algorithm.GeneticAlgorithm.EvaluateFitnessFunction(BC.Fitness_Function_ID[0],PopulationInitial.Chromosome[IndividualNumber], BC.S_max[0], BC.a_0[0], BC.a_max[0], BC.Delta_a[0],MAT.C[0],MAT.m[0], BC)
                    NumberOfEvaluations = NumberOfEvaluations +1
                    
                    PopulationCurrent.Chromosome[IndividualNumber] = PopulationInitial.Chromosome[IndividualNumber]
                    
                    # For comparison at the end of the optimisation, the fitness values for the initial population are also stored 
                    PopulationInitial.Fitness[IndividualNumber] = PopulationCurrent.Fitness[IndividualNumber]
    
                    # Extract Genotypes of each individual in the initial population
                    PopulationInitial.Genotype[IndividualNumber] = genetic_algorithm.Population.ExtractGenotype(PopulationInitial.loc[IndividualNumber,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
                    PopulationCurrent.Genotype[IndividualNumber] = genetic_algorithm.Population.ExtractGenotype(PopulationInitial.loc[IndividualNumber,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
    
                    # Store information into the PopulationComposition dictionary for current individual
                    PopulationComposition['Gen '+str(0)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(0)][0], Operation = "Initialization", Chromosome = PopulationInitial.Chromosome[IndividualNumber] , Fitness = PopulationInitial.Fitness[IndividualNumber], Pp = None, Parents = 0, FatigueCalculations = FatigueCalculations,ExperimentNumberID = ExperimentNumberID)
                    PopulationComposition['Gen '+str(Generation)][4] = PopulationCurrent # the offspring of the previous generation is the current population for this generation.
                    PopulationComposition['Gen '+str(Generation)][5] = np.mean(PopulationComposition["Gen "+str(Generation)][4].Fitness) # Population Arithmetic Mean value
                    
            else:                       # else use the Current population that has been produced through previous generations
            
                PopulationCurrent = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])
                
                for IndividualNumber in range(1,int(BC.N_pop)+1):
                                
                    PopulationCurrent.Fitness[IndividualNumber], FatigueCalculations = genetic_algorithm.GeneticAlgorithm.EvaluateFitnessFunction(BC.Fitness_Function_ID[0],PopulationOffspring.Chromosome[IndividualNumber], BC.S_max[0], BC.a_0[0], BC.a_max[0], BC.Delta_a[0],MAT.C[0],MAT.m[0], BC)
                    NumberOfEvaluations = NumberOfEvaluations +1

                    PopulationCurrent.Chromosome[IndividualNumber] = PopulationOffspring.Chromosome[IndividualNumber]
                    PopulationCurrent.Genotype[IndividualNumber] = genetic_algorithm.Population.ExtractGenotype(PopulationCurrent.loc[IndividualNumber,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
    
                    
                    #Store information into the PopulationComposition dictionary
                    PopulationComposition['Gen '+str(Generation)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation)][0], Operation = "Evaluation", Chromosome = PopulationCurrent.Chromosome[IndividualNumber] , Fitness = PopulationCurrent.Fitness[IndividualNumber], Pp = None, Parents = 0, FatigueCalculations = FatigueCalculations, ExperimentNumberID = ExperimentNumberID)
                    
                    PopulationComposition['Gen '+str(Generation)][4] = PopulationCurrent # the offspring of the previous generation is the current population for this generation.
                    PopulationComposition['Gen '+str(Generation)][5] = np.mean(PopulationComposition["Gen "+str(Generation)][4].Fitness) # Population Arithmetic Mean value


                 
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            
            Step 2.a Termination Condition - check whether the termination condition has been triggered 
            
            """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            
            TerminationCondition = genetic_algorithm.GeneticAlgorithm.CheckTermination(BC.NumberOfGenerations[0], Generation, PopulationCurrent)
            
            if TerminationCondition == True:
                                    
                """
                Extract the genotypes of all final chromosomes and add them to the PopulationComposition dictionary for analysis. 
                Calculate PopulationGeneComposition array for analysis.
                """
                
                for UniqueChromosome in range(0,len(PopulationComposition['Gen '+str(Generation)][0])):
                    
                    PopulationComposition['Gen '+str(Generation)][0].Genotype[UniqueChromosome] = genetic_algorithm.Population.ExtractGenotype(PopulationComposition['Gen '+str(Generation)][0].loc[UniqueChromosome,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
                        
                # Calculate and Store GeneComposition for the current generation
                
                PopulationComposition['Gen '+str(Generation)][1] = genetic_algorithm.Population.CalculateAlleleFrequenciesFromRelations(PopulationComposition['Gen '+str(Generation)][0], BC.T_dict[0], BC.n_total[0])
                PopulationComposition['Gen '+str(Generation)][2] = genetic_algorithm.Population.ConstructRelativeAlleleStrengthComposition(PopulationCurrent, ExperimentNumberID)
    
                
                # Create dataframe for final population, for easy reference
                PopulationFinal = PopulationCurrent
                
                # Extract Genotypes for the individuals in the final population dataframe (not most efficient, could fix this later)
                for UniqueChromosome in range(1,len(PopulationFinal)+1):
    
                    PopulationFinal.Genotype[UniqueChromosome] = genetic_algorithm.Population.ExtractGenotype(PopulationFinal.loc[UniqueChromosome,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
                      
                                        
                break
    
            
            else: 
                # Store information on the current population for the family tree
                
                """
                Transfer memory of unique chromosomes, fitness and genotypes to the next generation
                """
                
                PopulationComposition = genetic_algorithm.Population.TransferPopulation(Generation, PopulationComposition)
                
                     
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                
                Step 3. Selection - select the parents for crossover
                
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                
                if BC.Crossover[0] == str(True):
                    
                    PopulationOffspring = database_connection.Database.RetrievePopulationDataframe(BC.N_pop[0])

                    while PopulationOffspring.tail(1).Chromosome[BC.N_pop[0]] is None: #keep recombining until the entire PopulationOffspring is filled with offspring solutions
                        
                        # Return Selected Parent 1 and Parent 2 depending on selection heuristic. Input is the current population and necessary parameters

                        ParentSelected1, ParentSelected2 = genetic_algorithm.GeneticAlgorithm.SelectReproducingParents(PopulationCurrent, BC.SelectionOperator[0], BC)
                        
                        # Recombine the selected parents based on the crossover heuristic
                        
                        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                
                        Step 4. Crossover - recombine the selected parent solutions using the crossover heuristic specified
                        
                        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        
                        PopulationOffspring, Child1, Child2 = genetic_algorithm.GeneticAlgorithm.RecombineParents(PopulationCurrent.Chromosome[ParentSelected1], PopulationCurrent.Chromosome[ParentSelected2], PopulationOffspring, BC.Pc[0], BC.W[0], BC.CrossoverOperator[0], CONSTRAINTS, BC.n_total[0])
                        
                        # Store the offspring relations in the PopulationComposition dictionary
    
                        Children = [Child1, Child2]
                        for Child in range(0,len(Children)):
                            PopulationComposition['Gen '+str(Generation+1)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation+1)][0], Operation = "Crossover", Chromosome = Children[Child] , Fitness = 0, Pp = None, Parents = [PopulationCurrent.Chromosome[ParentSelected1],PopulationCurrent.Chromosome[ParentSelected2]],FatigueCalculations = 0,ExperimentNumberID = ExperimentNumberID)
        

                else: # if Crossover is not enabled, the surviving population becomes the offspring population. Bart: The size still needs to increase with a size, depending on Rs!
                    
                    PopulationCurrentSelectedPartial = genetic_algorithm.GeneticAlgorithm.SelectSurvivingPopulation(PopulationCurrent, BC.Rs[0]) # only mutation, then the selected, or surviving, solutions are copied twice into the offspring population.
                    PopulationCurrentSelected = PopulationCurrentSelectedPartial.append(PopulationCurrentSelectedPartial, ignore_index = True)
                    PopulationCurrentSelected.index = np.arange(1,len(PopulationCurrentSelected)+1)
                    
                    PopulationOffspring = PopulationCurrentSelected # if crossover has been disabled, the surviving population becomes the offspring population
                        
                    # Store the offspring relations into the PopulationComposition dictionary
    
     
                """
                Data Storage Step - Extract the genotypes of all current chromosomes and add them to the PopulationComposition dictionary for analysis. 
                """
                
                for UniqueChromosome in range(0,len(PopulationComposition['Gen '+str(Generation)][0])):
                    
                    PopulationComposition['Gen '+str(Generation)][0].Genotype[UniqueChromosome] = genetic_algorithm.Population.ExtractGenotype(PopulationComposition['Gen '+str(Generation)][0].loc[UniqueChromosome,"Chromosome"] , BC.Delta_a[0], BC.n_total[0], BC.W[0])
                        
                """
                Data Storage Step - Store information on the composition of the current population in dictionary
                """                    
                    
                # Calculate and Store GeneComposition for the current generation  
                
                PopulationComposition['Gen '+str(Generation)][1] = genetic_algorithm.Population.CalculateAlleleFrequenciesFromRelations(PopulationComposition['Gen '+str(Generation)][0], BC.T_dict[0], BC.n_total[0])
                PopulationComposition['Gen '+str(Generation)][2] = genetic_algorithm.Population.ConstructRelativeAlleleStrengthComposition(PopulationCurrent, ExperimentNumberID) 
                PopulationComposition['Gen '+str(Generation+1)][3] = NumberOfEvaluations 


                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
                Step 5. Mutation - mutate the offspring population using the mutation heuristic specified
                
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 
               
                if BC.Mutation[0] == str(True):
                                            
                    for IndividualNumber in range(1,len(PopulationOffspring)+1):
                        #print("Starting mutation of the Offspring Population for Individual...", IndividualNumber)

                        ParentChromosome = PopulationOffspring.Chromosome.loc[IndividualNumber].copy() #removed .copy()
                        
                        RelativeStrengthOverFrequency = PopulationComposition["Gen "+str(Generation)][2] / PopulationComposition["Gen "+str(Generation)][1]
                        
                        ChildChromosome = genetic_algorithm.GeneticAlgorithm.MutateChromosome(PopulationOffspring.Chromosome.loc[IndividualNumber], BC.MutationOperator[0], BC.Pm[0], BC.n_total[0], BC.W[0],  BC.T_dict[0], CONSTRAINTS, RelativeStrengthOverFrequency, BC.Delta_a[0], PopulationOffspring, PopulationOffspring.index[IndividualNumber-1], PopulationComposition['Gen '+str(Generation+1)][0])
                        
                        PopulationOffspring.Chromosome[IndividualNumber] = ChildChromosome
                        
                        Identical = None
                        if np.array_equal(ParentChromosome.Thickness,PopulationOffspring.Chromosome.loc[IndividualNumber].Thickness):
                            Identical = True
                        
                        else:
                            Identical = False
                            
                        #print(Identical)
                        """
                        Parent_ID from Offspring Population is not the same as Parent_ID from PopulationComposition!!
                        """
                        # Store the relations after mutating into the PopulationComposition dictionary
    
                        PopulationComposition['Gen '+str(Generation+1)][0] = genetic_algorithm.Population.StorePopulationData(PopulationDataframe = PopulationComposition['Gen '+str(Generation+1)][0], Operation = "Mutation", Chromosome = ChildChromosome, Fitness = 0, Pp = None, Parents = [ParentChromosome],FatigueCalculations = 0,ExperimentNumberID = ExperimentNumberID)
        
            
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
        
        PopulationFinal = PopulationFinal.sort_values("Fitness", ascending= False, kind='mergesort')
        
        # Show top 3 crenellation patterns in the Final Population
        
        topPatterns = visuals.PopulationVisuals.ShowTop3CrenellationPatterns(PopulationFinal, PopulationInitial, ExperimentNumberID)
        
        # Show Allele strength development over generations
        
        alleleStrength = 0 #= visuals.PopulationVisuals.ShowAlleleStrengthComposition(PopulationComposition,  ExperimentNumberID) show this again!
        
        # Show GeneComposition for intervals of generations
        
        ##visuals.PopulationVisuals.ShowPopulationConvergence(PopulationComposition, BC.NumberOfGenerations[0])
        
        # NeutralAlleleStrengths = visuals.PopulationVisuals.ShowNeutralAlleles(PopulationComposition, BC.Fitness_Function_ID[0],BC.NumberOfGenerations[0],ExperimentNumberID) show this again!
        
        """
        Store the GA PopulationComposition in a pickle per ExperimentID
        """

        PopulationCompositionComplete = database_connection.Database.StorePopulationCompositionDictionary(PopulationComposition, ExperimentNumberID)
        
        """  
        #==============================================================================
        #                          Runtime of the algorithm
        #==============================================================================
        """
        time_elapsed = (time.clock() - time_start)
        print("Calculation time:", round(time_elapsed,2), "seconds, or ",round(time_elapsed/60,2)," minutes")
        

        return PopulationComposition, topPatterns, alleleStrength, PopulationFinal