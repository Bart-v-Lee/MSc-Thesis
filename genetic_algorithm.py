#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:13:43 2017

@author: Bart
"""
import pandas as pd
import numpy as np
from crenellation import crenellation
import sys


class genetic_algorithm:

    def __init__(self, bc, material):
        self.bc = bc
        self.material = material


    
        
    def children_population(self, bc):
        """
        Constructs an empty dataframe as the children population. 
        """
        pop_size = int(bc.ix["Population size"])
        array = np.zeros((pop_size,7))
        index = range(1,pop_size+1)
        list = {"Original Indi. No","Fitness", "Chromosome", "Cren Design", "Balance", "Lower Bound","Upper Bound"}
        population_children = pd.DataFrame(data=array, index = index, columns = list, dtype = 'object')
                
        for i in range(1,pop_size+1):
            population_children["Original Indi. No"][i] = index[i-1]
        
        return population_children
        
                

        


    def rank_parents(self, bc, population_eval):
        
        ranking_method = bc.ix["Ranking Method"]
        """
        Choose selection method of parents based on condition provided in the boundary conditions bc
        """
        
        if ranking_method == 'Rank':
            population_ranked = genetic_algorithm.fitness_ranking_method(self,bc, population_eval)
            
        elif ranking_method == 'Relative Fitness':
            pass
            
        elif ranking_method == 'Inverse Rank':
            population_ranked = genetic_algorithm.inverse_ranking_method(self, bc, population_eval)
            
        elif ranking_method == 'Tournament': #not developed yet at this point
            population_ranked = genetic_algorithm.tournament_method(self)
            
    
        return population_ranked
            
    """           
    #==============================================================================
    #                       Ranking methods
    #==============================================================================
    """

    def fitness_ranking_method(self, bc, population_eval):
        """
        Determines the rank of individuals from high to low fitness
        """
        
        selection_rate = bc.ix["Selection Rate"]
        pop_size = int(bc.ix["Population size"])
        population_ranked = population_eval.sort_values("Fitness",ascending=False, kind='mergesort')
        cut_off_position = int(selection_rate * pop_size)
        population_selected = population_ranked[:cut_off_position]
    
        return population_selected
    
    def inverse_ranking_method(self, bc, population_eval):
        
        """
        Determines the rank of individuals from low to high fitness
        """
        
        selection_rate = bc.ix["Selection Rate"]
        pop_size = int(bc.ix["Population size"])
        population_ranked = population_eval.sort_values("Fitness",ascending=True, kind='mergesort')
        cut_off_position = int(selection_rate * pop_size)
        population_selected = population_ranked[:cut_off_position]
        
        population_selected = population_selected.sort_values("Fitness", ascending=False, kind = 'mergesort')
        
        return population_selected
        
        
    def tournament_method():
        pass
    
    
    def pairing_probability(self, bc, population_children, population_parents):
        """
        Calculates the probability of an individual being chosen during pairing of the parents
        """
        pairing_set = bc.ix["Pairing Set"]
        
        if pairing_set == "Rank Weighting":
               
            pop_size_selected = len(population_parents)
            
            population_parents.index = range(1,pop_size_selected+1)

            population_parents["Pm"] = (pop_size_selected - population_parents.index + 1) / (sum(range(1,pop_size_selected+1)))
            population_parents["Pm cumulative"] = population_parents["Pm"].cumsum()
            
            
        elif pairing_set == "Fitness Weighting": #possibility to develop
            pass
        
        elif pairing_set == "Tournament schema": #possibility to develop
            pass
        

        # print(population_parents)
        
        return population_parents

    """
    #==============================================================================
    #                       Recombination 
    #==============================================================================
    """  
    def recombination(self, bc, material, population_selected):
        
        population = genetic_algorithm(bc,material)
        
        recombination_criteria = bc.ix["Recombination Method"]

        """
        Cross-over of selected parents
        """
        if recombination_criteria == "Set 1":
            population_children, number_of_elites = genetic_algorithm.cross_over(self, bc, material, population_selected)
            
        elif recombination_criteria == "Set 2":
            population_children, number_of_elites = genetic_algorithm.uniform_cross_over(self)
            
        elif recombination_criteria == "Set 3":
            population_children, number_of_elites = genetic_algorithm.initialize_population(self, bc,material,population)
            
        elif recombination_criteria == "Set 4":
            population_children, number_of_elites = genetic_algorithm.redistribution(self)
            
        else:
            pass

        """
        Mutation of children
        """
        
        mutation_criteria = bc.ix["Mutation Set"]    
            
        if mutation_criteria == "Set 1":
            population_children = genetic_algorithm.mutate(self, bc, population_children, number_of_elites)
            
        if mutation_criteria == "Set 2":
            population_children = genetic_algorithm.mutate_swap(self, bc, population_children, number_of_elites)
        
        if mutation_criteria == "Set 3":
            population_children = genetic_algorithm.mutate_swap_ref_5_10cont_8thick(self, bc, population_children, number_of_elites)

        else:
            pass
        
        #print(population_children)
        return population_children
        
    """
    #==============================================================================
    #                        Cross over methods
    #==============================================================================
    """    
    def cross_over(self, bc, material, population_selected):
        
        elitism_settings = bc.ix["Elitism"]
        
        if elitism_settings == "True":
            population_children, number_of_elites = genetic_algorithm.apply_elites(self,bc,population_selected)
            print("Elitism has been used")
            
        elif elitism_settings == "Inverse":
            population_children = genetic_algorithm.apply_elites_inverse(self,bc,population_selected)
            
        else:
            population_children = genetic_algorithm.children_population(self, bc)
            number_of_elites = 0
        
        cross_over_method = bc.ix["Cross-over Method"]
        population_parents = population_selected
        population_parents = genetic_algorithm.pairing_probability(self, bc, population_children, population_parents)
        
        if cross_over_method == "Single Point":
            population_children = genetic_algorithm.single_point_crossover(self, bc, population_children, population_parents)
        
        elif cross_over_method == "Addition":
            population_children = genetic_algorithm.addition_crossover(self, bc, population_children, population_parents)

        return population_children, number_of_elites

        
        
    def pair_parents(self, bc, population_parents):
        """
        Picks a float value between 0.0 and 1.0
        """
        pair_parameter = np.random.uniform(0.0,1.0)
        """
        Looks up the chosen individual in the cumulative probability distribution of all parent individuals
        """
        number_of_true = population_parents["Pm cumulative"][population_parents["Pm cumulative"]>= pair_parameter].count()
        parent_index = (len(population_parents) - number_of_true)+1
        """
        Returns the index of the chosen parent in the parent population
        """
        return parent_index
        

    def single_point_crossover(self, bc, population_children, population_parents):
        """
        Assign & calculate variables
        """
        pop_size = bc.ix["Population size"]
        t_ref = bc.ix["Reference thickness"]
        number_of_containers = bc.ix["number_of_containers"] #fill this in into GA boundary conditions
        half_width = bc.ix["Width"]/2
        output_children_per_couple = 2
        number_of_elites = int(pop_size -  population_children["Chromosome"].count())
        number_of_children = int(pop_size - number_of_elites) #- np.count_nonzero(population_children["Chromosome"])
        number_of_couples = int(number_of_children / output_children_per_couple)
        chromosome_half_length = int(len(population_parents["Chromosome"][1])/2)
        container_width = chromosome_half_length / number_of_containers
        
        """
        Pair two parents and calculate the array of feasible crossover points
        """
        
        for i in range(1+number_of_elites,number_of_couples+1):
            feasible_crossover_points = []

            while feasible_crossover_points == []:
                parent_1_index = genetic_algorithm.pair_parents(self, bc,population_parents) #chooses parent 1
                parent_2_index = genetic_algorithm.pair_parents(self, bc,population_parents) #chooses parent 2
                
                while parent_1_index == parent_2_index:
                    parent_2_index = genetic_algorithm.pair_parents(self, bc,population_parents) #ensures that the parents are not the same
                                
                for crossover_point in range(1,chromosome_half_length-1):
    
                    area_parent_1 = np.sum(population_parents["Chromosome"][parent_1_index][:crossover_point])
                    area_parent_2 = np.sum(population_parents["Chromosome"][parent_2_index][:crossover_point])
                    
                    """
                    Check whether the area of both parents' chromosomes until the crossover point is within narrow range of each other.
                    If so, the crossover point is feasible as it will yield children of equal area compared to its parents.
                    """
                    
                    if area_parent_1 - 0.0001 < area_parent_2 < area_parent_1 +0.0001: 
                        
                        container_edges = np.arange(1,number_of_containers)*container_width
                        
                        """
                        If the crossover point is at an edge of the container, add it to the array of feasible crossover points.
                        Not most computationally efficient, yet not a bottleneck.
                        """
                        
                        if crossover_point in container_edges:
                            feasible_crossover_points = np.append(feasible_crossover_points,int(crossover_point))
                            print("possible crossover points",feasible_crossover_points)
    #                        print("area parent 1",area_parent_1)
    #                        print("area parent 2", area_parent_2)
            
#            print("total feasible crossover points",feasible_crossover_points)

            """
            From array of feasible crossover points, choose one crossover point following a uniform random distribution
            """
            cross_over_point = np.random.choice(feasible_crossover_points)
#            print("crossover point chosen", cross_over_point)

            """
            Perform the crossover by combining different parts of both parents.
            """
            parent_1_left = population_parents["Chromosome"][parent_1_index][:cross_over_point]
            parent_2_left = population_parents["Chromosome"][parent_2_index][:cross_over_point]
            parent_1_right = population_parents["Chromosome"][parent_1_index][cross_over_point:chromosome_half_length]
            parent_2_right = population_parents["Chromosome"][parent_2_index][cross_over_point:chromosome_half_length]
            child_1_chromosome_left_half = np.append(parent_1_left, parent_2_right)
            child_2_chromosome_left_half = np.append(parent_2_left, parent_1_right)
            child_1_chromosome = np.append(child_1_chromosome_left_half,np.flipud(child_1_chromosome_left_half))
            child_2_chromosome = np.append(child_2_chromosome_left_half,np.flipud(child_2_chromosome_left_half))
            """
            Calculate and apply the rebalancing of the new chromosome
            """
#            individual_no = i
#            individual_no_2 = i + number_of_couples
            
#            child_1_chromosome = crenellation.apply_balance_crossover(self, child_1_chromosome, bc, individual_no)
#            child_2_chromosome = crenellation.apply_balance_crossover(self, child_2_chromosome, bc, individual_no_2)
            """
            Assign new children to the children population
            """
            population_children["Chromosome"][i] = child_1_chromosome
            population_children["Chromosome"][i+number_of_couples] = child_2_chromosome
            
#            """
#            Check whether the area of the children is equal to the parents
#            """
#            area_child_1 = int(np.sum(child_1_chromosome))
#            area_child_2 = int(np.sum(child_2_chromosome))
#            area_parent_1 = np.sum(population_parents["Chromosome"][parent_1_index])
#            area_parent_2 = np.sum(population_parents["Chromosome"][parent_2_index])
#            area_ref = int(t_ref * half_width)
            
#            if area_child_1 not in range(int(area_ref - 10), int(area_ref +10)) or area_child_2 not in range(int(area_ref - 10), int(area_ref +10)):
#                print("area out of bounds after crossover for children individual ", i," or ",i+number_of_couples)
#                print("the area for child 1 was ",area_child_1," and child 2 was ",area_child_2)
#                print("the area for parent 1 was ",area_parent_1," and parent 2 was ",area_parent_2)
#                print("Population parents was ",population_parents)
#                print("parent 1 was number ",parent_1_index," and parent 2 was number ",parent_2_index)
#                print("Population children became ", population_children)
#                sys.exit('GA stopped due to unequal area with reference panel after crossover')
#            
        return population_children
        
    
    def uniform_cross_over(self):
        pass
    
    def addition_crossover(self, bc, population_children, population_parents):
        """
        Currently not in use
        """
        
        t_ref = bc.ix["Reference thickness"]
        pop_size = bc.ix["Population size"]
        output_children_per_couple = 1
        number_of_children = pop_size #- np.count_nonzero(population_children["Chromosome"])
        number_of_couples = int(number_of_children / output_children_per_couple)
        chromosome_half_length = int(len(population_parents["Chromosome"][1])/2)
        
        for i in range(1,number_of_couples+1):
            
            parent_1_index = genetic_algorithm.pair_parents(self, bc,population_parents)
            parent_2_index = genetic_algorithm.pair_parents(self, bc,population_parents)
            
            while parent_1_index == parent_2_index:
                parent_2_index = genetic_algorithm.pair_parents(self, bc,population_parents)
            
            parent_1 = population_parents["Chromosome"][parent_1_index][:]
            parent_2 = population_parents["Chromosome"][parent_2_index][:]
            child_1_chromosome = (parent_1 + parent_2) / 2
            """
            Calculate and apply the rebalancing of the new chromosome
            """
            population_children["Chromosome"][i] = child_1_chromosome
        
        return population_children
        
    """
    #==============================================================================
    #                           Mutation methods        
    #==============================================================================
    """
            
    def mutate(self, bc, population_children, number_of_elites):
        """
        Mutate for highly refined crenellation patterns, not the simple cases. Only review if necessary at this point.
        """
        #print(number_of_elites)
        t_ref = bc.ix["Reference thickness"]
        pop_size_all = bc.ix["Population size"]
        pop_size_non_elites = pop_size_all - number_of_elites
        mutation_rate = bc.ix["Mutation Rate"]
        chromosome_length = int(len(population_children["Chromosome"][1])/2)
        mutation_width = bc.ix["Mutation Width"]
        number_of_mutations = int((mutation_rate * pop_size_non_elites * chromosome_length)/(mutation_width*chromosome_length))  #subtract the elites
        total_locations = int(pop_size_all * chromosome_length)
        
        start_mutation_location =int(number_of_elites * chromosome_length)
        
        mutation_locations = np.random.randint(start_mutation_location,total_locations, number_of_mutations)
        
        print("Starting mutation of children")
        for i in range(0,number_of_mutations):
            """
            Pick one of the mutation locations and find the respective chromosome
            """
            individual_no = int(np.floor(mutation_locations[i] / chromosome_length))
            mutation_location = mutation_locations[i] - int(individual_no * chromosome_length)
            individual_chromosome = population_children["Chromosome"][individual_no+1]
            """
            Expand the number of containers subjected to the mutation operation
            """
            bandwidth_left = int(max(0,mutation_location - mutation_width * chromosome_length))
            bandwidth_right = int(min(chromosome_length,mutation_location + mutation_width * chromosome_length))
            thickness_left = individual_chromosome[bandwidth_left]
            thickness_right = individual_chromosome[bandwidth_right]
            """
            Calculate the A_balance of the existing section of the chromosome
            """
            current_balance = np.sum((individual_chromosome[bandwidth_left:bandwidth_right+1]) - t_ref)
            """
            Re-initialize crenellation pattern for the mutation range
            """
            individual_chromosome, mutated_balance,t = crenellation.rand_thickness_mutation(self, individual_chromosome, bandwidth_left, bandwidth_right, thickness_left, thickness_right, bc)
            """
            Apply balance for the mutation range
            """
            individual_chromosome = crenellation.apply_balance_mutation(self, t, bandwidth_left, bandwidth_right, current_balance, mutated_balance, individual_chromosome, bc)
            """
            Mirror the mutated region of the chromosome to the right part of the chromosome
            """
            individual_chromosome_left_half = individual_chromosome[:chromosome_length]
            individual_chromosome = np.append(individual_chromosome_left_half,np.flipud(individual_chromosome_left_half))
            """
            Insert mutated chromosome back into the population
            """
            population_children["Chromosome"][individual_no+1] = individual_chromosome
            
        return population_children

    def mutate_swap(self, bc, population_children, number_of_elites):
        """
        Mutation through swapping of material for highly refined crenellation patterns. Only review if necessary.
        """
        t_ref = bc.ix["Reference thickness"]
        pop_size = bc.ix["Population size"]
        mutation_rate = bc.ix["Mutation Rate"]
        chromosome_length = int(len(population_children["Chromosome"][1])/2)
        number_of_mutations = int(mutation_rate * pop_size * chromosome_length)  #subtract the elites
        total_locations = int(pop_size * chromosome_length)
        
        mutation_locations = np.random.randint(0,total_locations, number_of_mutations)
        
        #print("Starting mutation of children")
        for i in range(0,number_of_mutations):
            """
            Pick one of the mutation locations and find the respective chromosome
            """
            individual_no = int(np.floor(mutation_locations[i] / chromosome_length))
            individual_location = mutation_locations[i] - int(individual_no * chromosome_length)
            individual_chromosome = population_children["Chromosome"][individual_no+1]
            """
            Swap thickness with another location in the chromosome
            """
            individual_chromosome = crenellation.swap_thickness_mutation(self, individual_chromosome, individual_location, bc)
            """
            """
            individual_chromosome_left_half = individual_chromosome[:chromosome_length]
            individual_chromosome = np.append(individual_chromosome_left_half,np.flipud(individual_chromosome_left_half))
            """
            Insert mutated chromosome back into the population
            """
            population_children["Chromosome"][individual_no+1] = individual_chromosome
            
        return population_children

    def mutate_swap_ref_5_10cont_8thick(self, bc, population_children, number_of_elites):
        """
        Mutate through swapping material between containers for crenellation patterns from reference paper. 
        """
        
        #print(population_children)
        t_ref = bc.ix["Reference thickness"]
        delta_t_min = bc.ix["Layer thickness"]
        mutation_rate = bc.ix["Mutation Rate"]
        number_of_containers = bc.ix["number_of_containers"]
        population_size = bc.ix["Population size"]
        half_width = bc.ix["Width"]/2
        area_ref = t_ref * half_width
        container_width = (half_width /2) / number_of_containers
        number_of_mutations = int(number_of_containers * population_size * mutation_rate)
        thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}

        """
        Choose which containers should be mutated, and mutate them by the minimum thickness level. Direction is chosen randomly uniform.
        """
        for i in range(1, number_of_mutations):
            mutate_location = int(np.random.choice(np.arange(1,population_size*number_of_containers)))
            individual_number = int(np.ceil(mutate_location / number_of_containers))
            area_chromosome_old = np.sum(population_children.Chromosome[individual_number])

            #print("individual no ",individual_number)
            
            mutate_local_loc_1 = mutate_location - ((individual_number-1)*number_of_containers)
            mutate_local_loc_2_options = np.delete(np.arange(1,number_of_containers+1),mutate_local_loc_1-1)
            mutate_local_loc_2 = np.random.choice(mutate_local_loc_2_options)
            container_1_range_chromosome = [int(((mutate_local_loc_1-1) * container_width)),int((mutate_local_loc_1 * container_width))]
            container_2_range_chromosome = [int(((mutate_local_loc_2-1) * container_width)),int((mutate_local_loc_2 * container_width))]
            """
            Chosing maximum transfer of thickness between the two chosen containers
            """
            current_thickness_1 = population_children.Chromosome[individual_number][container_1_range_chromosome[0]] #hier gaat iets fout met current thickness
            #print("current thickness" , current_thickness_1)
            current_thickness_2 = population_children.Chromosome[individual_number][container_2_range_chromosome[0]]
            
            available_thickness_increase = int((thickness_dict[max(thickness_dict)] - current_thickness_1)/ delta_t_min) #calculating how much a container thickness can increase until it reaches the maximum thickness level
            available_thickness_decrease = int((current_thickness_2 - thickness_dict[min(thickness_dict)]) / (delta_t_min)) #give some margin within the rounding, otherwise will give an error
            available_thickness_change = min(available_thickness_increase, available_thickness_decrease) #if decrease is 0 , then something goes wrong
            
            current_thickness_level_1 = int((current_thickness_1 - thickness_dict[min(thickness_dict)] ) / (delta_t_min - 0.01))
            
            if available_thickness_change == 0:
                continue
            else:
                thickness_options_array = np.arange(current_thickness_level_1, current_thickness_level_1 + available_thickness_change +1)
            
            """
            Applying the thickness changes to the respective containers
            """
            new_thickness_no_1 = np.random.choice(thickness_options_array)
            new_thickness_level_1 = thickness_dict[new_thickness_no_1]
            #thickness_difference = new_thickness_level - population_children.Chromosome[individual_number][container_1_range_chromosome[0]]
            thickness_level_difference = new_thickness_no_1 - current_thickness_level_1
            new_thickness_level_2 = current_thickness_2 - (delta_t_min*thickness_level_difference)
            
            population_children.Chromosome[individual_number][container_1_range_chromosome[0]:container_1_range_chromosome[1]] = new_thickness_level_1
            population_children.Chromosome[individual_number][container_2_range_chromosome[0]:container_2_range_chromosome[1]] = new_thickness_level_2

            """
            Mirror mutated chromosome around centre to make it symmetrical again
            """
            population_children.Chromosome[individual_number] = np.append(population_children.Chromosome[individual_number][:(half_width/2)],np.flipud(population_children.Chromosome[individual_number][:(half_width/2)]))
            
            if abs(new_thickness_level_1 - current_thickness_1) > 0.03 and new_thickness_level_2 == current_thickness_2 or new_thickness_level_1 == current_thickness_1 and abs(new_thickness_level_2 - current_thickness_2) > 0.03:
                print("thickness out of bounds for individual", individual_number)
                print("thickness 1 for this individual was ", current_thickness_1," at current thickness level ", current_thickness_level_1," and now it is ", new_thickness_level_1)
                print("the thickness options were ", thickness_options_array)
                print("thickness 2 for this individual was ", current_thickness_2, " and now it is ", new_thickness_level_2)
                print("mutation took place at location ",mutate_local_loc_1, "and mirrored at location ", mutate_local_loc_2)
                sys.exit('GA stopped due to unequal area with reference panel after mutation')
            
        #print(population_children)
        return population_children
        
        
    """
    #==============================================================================
    #                           Elitism methods
    #==============================================================================
    """
            
            
    def apply_elites(self, bc, population_selected):
        
        """
        Take children population and fill in first individuals with elites
        """
        
        pop_size = int(bc.ix["Population size"])
        population_children = genetic_algorithm.children_population(self, bc)
        
        elites_perc = bc.ix["Elites percentage"]
        number_of_elites = int(pop_size * elites_perc)
        
        if pop_size * elites_perc < 0:
            print("No Elites chosen due to wrong elites percentage versus population size settings")
        
        else:
            for k in range(1,number_of_elites+1):
                elite_index = population_selected.index[k]
                population_children.at[k,"Chromosome"] = population_selected["Chromosome"][elite_index]
            
                #print("Population with elites applied",population_children)
        
        return population_children, number_of_elites
        
             
    def apply_elites_inverse(self, bc, population_selected):
        
        """
        Take children population and fill in first individuals with elites
        """
        
        pop_size = int(bc.ix["Population size"])
        population_children = genetic_algorithm.children_population(self, bc)
        population_selected_inverse = population_selected.sort_values("Fitness",ascending=True, kind='mergesort')
        
        elites_perc = bc.ix["Elites percentage"]
        number_of_non_elites = int(pop_size * (1-elites_perc))
        #print(number_of_non_elites)
        
        if pop_size * elites_perc < 0:
            print("No Elites chosen due to wrong elites percentage versus population size settings")
        
        for k in range(0,number_of_non_elites-1):
            elite_index = population_selected_inverse.index[k]
            population_children.at[k,"Chromosome"] = population_selected_inverse["Chromosome"][elite_index]
            
            
        #print("Population with elites applied",population_children)
        
        return population_children
        
        
    """
    #==============================================================================
    #                    Population convergence methods
    #==============================================================================
    """
    """
    Stores key data per generation for analysis
    """

    def population_convergence(self, bc, population_eval,g, convergence_overview):
        population_convergence = convergence_overview
        number_of_generations = int(bc.ix["Number of Generations"])
        array = np.zeros(((number_of_generations),4))
        index = range(1,(number_of_generations)+1)
        ranked_population = population_eval.sort_values("Fitness", ascending=False,kind='mergesort')

        if g == 0:
            list = {"Individual No.", "Crenellation Pattern","Fitness", "Generation No"}
            population_convergence = pd.DataFrame(data=array, index = index, columns = list, dtype = 'object')
        else:
            population_convergence["Fitness"][g] = ranked_population["Fitness"].head(1).values        #change the number for the number of individuals you want to show in the convergence
            population_convergence["Crenellation Pattern"][g] = ranked_population["Chromosome"].head(1).values           
            population_convergence["Individual No."][g] =  ranked_population["Original Indi. No"].head(1).values         
            population_convergence["Generation No"][g] =  g  
            
        return population_convergence
        
        
