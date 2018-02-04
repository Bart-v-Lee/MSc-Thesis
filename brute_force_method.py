#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:45:20 2017

@author: Bart van der Lee
"""

"""
This file can be used to brute force calculate the objective function for all possible solutions 
in the searching space. As a result, the complete fitness landscape can be modeled.
"""


"""
Step 1. Provide constraints & boundary conditions for the crenellation pattern searching space
Step 2. Calculate the total area of the plate for each unique crenellation pattern
Step 3. If the area is not equal to area_reference, the pattern is not possible
Step 4. Calculate the fatigue life for the remainder of unique crenellation patterns
Step 5. 
Step 6.
"""

"""
Step 1. Provide constraints & boundary conditions for the crenellation pattern searching space
"""
import numpy as np
import pandas as pd

class brute_force_method:

    def __init__(self,bc):
        self.bc = bc

    def brute_force_method_simple(self,bc):
        population = []
        t_min = bc.ix["minimum thickness"]
        t_ref = bc.ix["Reference thickness"]
        delta_x = bc.ix["Stepsize horizontal"]
        half_width = bc.ix["Width"]/2
        number_of_containers = 2
        container_width = half_width / number_of_containers
        area_ref = float(half_width * t_ref)
        
        thickness_dict = {0: 2 ,1: 2.33 , 2: 2.67 , 3: 3, 4: 3.33, 5: 3.67, 6: 4}
            
        for thickness_container_1 in range(0,7):
                
            for thickness_container_2 in range(0,7):

                area = thickness_dict[thickness_container_1]*container_width + thickness_dict[thickness_container_2]*container_width 
                if area == area_ref:
                    chromosome_cont1 = np.repeat(thickness_dict[thickness_container_1],container_width/delta_x)
                    chromosome_cont2 = np.repeat(thickness_dict[thickness_container_2], container_width/delta_x)
                    chromosome_left = np.append(chromosome_cont1, chromosome_cont2)

                    chromosome = [chromosome_total , area]
                    population.append(chromosome)
                    
                else:
                    continue
        
        population = np.array(population)
        index = list(range(1,len(population)+1))
        columns = {"Chromosome", "Area", "Fitness","Original Indi. No", "Cren Design", "Balance", "Lower Bound", "Upper Bound"}
        data = np.zeros((len(population),len(columns)), dtype = 'float')
        population_df = pd.DataFrame(data = data, index = index, columns = columns)
        
        population_df.Chromosome = population[:,0]
        population_df.Area = population[:,1]
        
        
        return population_df
        
    
        
    def brute_force_method_custom_refinement(self):
        population = []
        t_min = 2
        t_ref = 3
        width = 150
        number_of_containers = 5
        container_width = width / number_of_containers
        area_ref = float(width * t_ref)
        
        thickness_dict = {0: 2 ,1: 2.33 , 2: 2.67 , 3: 3, 4: 3.33, 5: 3.67, 6: 4}
            
        for thickness_container_1 in range(0,7):
                
            for thickness_container_2 in range(0,7):
                
                for thickness_container_3 in range(0,7):
                    
                    area = thickness_dict[thickness_container_1]*container_width + thickness_dict[thickness_container_2]*container_width + thickness_dict[thickness_container_3]*container_width 
                    if area == area_ref:
                        chromosome = [thickness_dict[thickness_container_1], thickness_dict[thickness_container_2], thickness_dict[thickness_container_3], area]

                        population.append(chromosome)
                        
                    else:
                        continue
        
        number_of_unique_chromosome = len(population)
        
        return population_of_unique_chromosomes
    

    def brute_force_method_reference_5cont_8thick(self, bc):
        population = []
        t_min = bc.ix["minimum thickness"]
        t_ref = bc.ix["Reference thickness"]
        t_ref = 2.9
        delta_x = bc.ix["Stepsize horizontal"]
        half_width = bc.ix["Width"]/2
        number_of_containers = 5
        container_width = (half_width /2) / number_of_containers
        area_ref = int((half_width/2) * t_ref)
        
        thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}
            
        for thickness_container_1 in range(0,8):
                
            for thickness_container_2 in range(0,8):
                
                for thickness_container_3 in range(0,8):
                    
                    for thickness_container_4 in range(0,8):
                        
                        for thickness_container_5 in range(0,8):
                    
                            area = int(thickness_dict[thickness_container_1]*container_width + thickness_dict[thickness_container_2]*container_width + thickness_dict[thickness_container_3]*container_width + thickness_dict[thickness_container_4]*container_width + thickness_dict[thickness_container_5]*container_width)
                            if area in range(area_ref -3 , area_ref +3):
                                chromosome_cont1 = np.repeat(thickness_dict[thickness_container_1],container_width/delta_x)
                                chromosome_cont2 = np.repeat(thickness_dict[thickness_container_2], container_width/delta_x)
                                chromosome_cont3 = np.repeat(thickness_dict[thickness_container_3], container_width/delta_x)
                                chromosome_cont4 = np.repeat(thickness_dict[thickness_container_4], container_width/delta_x)
                                chromosome_cont5 = np.repeat(thickness_dict[thickness_container_5], container_width/delta_x)
                                chromosome_left = np.append(chromosome_cont1, [chromosome_cont2, chromosome_cont3, chromosome_cont4, chromosome_cont5] )
                                chromosome_total = np.append(chromosome_left, np.flipud(chromosome_left))
                                chromosome = [ chromosome_total , area]
                                population.append(chromosome)
                                
                            else:
                                continue
        
        population = np.array(population)
        index = list(range(1,len(population)+1))
        columns = {"Chromosome", "Area", "Fitness","Original Indi. No", "Cren Design", "Balance", "Lower Bound", "Upper Bound"}
        data = np.zeros((len(population),len(columns)), dtype = 'float')
        population_df = pd.DataFrame(data = data, index = index, columns = columns)
        
        population_df.Chromosome = population[:,0]
        population_df.Area = population[:,1]
        
        return population_df
    
    def brute_force_method_reference_5cont_8thick_stringers(self, bc):
        population = []
        t_min = bc.ix["minimum thickness"]
        t_ref = bc.ix["Reference thickness"]
        t_ref = 2.9
        delta_x = bc.ix["Stepsize horizontal"]
        half_width = bc.ix["Width"]/2
        number_of_containers = 5
        container_width = (half_width /2) / number_of_containers
        area_ref = int((half_width/2) * t_ref)
        
        thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}
            
        for thickness_container_1 in range(0,8):
                
            for thickness_container_2 in range(0,8):
                
                for thickness_container_3 in range(0,8):
                    
                    for thickness_container_4 in range(0,8):
                        
                        for thickness_container_5 in range(0,8):
                    
                            area = int(thickness_dict[thickness_container_1]*container_width + thickness_dict[thickness_container_2]*container_width + thickness_dict[thickness_container_3]*container_width + thickness_dict[thickness_container_4]*container_width + thickness_dict[thickness_container_5]*container_width)
                            if area in range(area_ref -3 , area_ref +3):
                                chromosome_cont1 = np.repeat(thickness_dict[thickness_container_1],container_width/delta_x)
                                chromosome_cont2 = np.repeat(thickness_dict[thickness_container_2], container_width/delta_x)
                                chromosome_cont3 = np.repeat(thickness_dict[thickness_container_3], container_width/delta_x)
                                chromosome_cont4 = np.repeat(thickness_dict[thickness_container_4], container_width/delta_x)
                                chromosome_cont5 = np.repeat(thickness_dict[thickness_container_5], container_width/delta_x)
                                chromosome_left = np.append(chromosome_cont1, [chromosome_cont2, chromosome_cont3, chromosome_cont4, chromosome_cont5] )
                                chromosome_total = np.append(chromosome_left, np.flipud(chromosome_left))
                                chromosome = [ chromosome_total , area]
                                population.append(chromosome)
                                
                            else:
                                continue
        
        population = np.array(population)
        index = list(range(1,len(population)+1))
        columns = {"Chromosome", "Area", "Fitness","Original Indi. No", "Cren Design", "Balance", "Lower Bound", "Upper Bound"}
        data = np.zeros((len(population),len(columns)), dtype = 'float')
        population_df = pd.DataFrame(data = data, index = index, columns = columns)
        
        population_df.Chromosome = population[:,0]
        population_df.Area = population[:,1]
        
        return population_df
    

    def brute_force_method_reference_10cont_8thick(self, bc):
        population = []
        t_min = bc.ix["minimum thickness"]
        t_ref = bc.ix["Reference thickness"]
        t_ref = 2.9
        delta_x = bc.ix["Stepsize horizontal"]
        half_width = bc.ix["Width"]/2
        number_of_containers = 10
        container_width = (half_width /2) / number_of_containers
        area_ref = int((half_width/2) * t_ref)
        
        thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}
            
        for thickness_container_1 in range(0,8):
                
            for thickness_container_2 in range(0,8):
                
                for thickness_container_3 in range(0,8):
                    
                    for thickness_container_4 in range(0,8):
                        
                        for thickness_container_5 in range(0,8):
                            
                            for thickness_container_6 in range(0,8):
                                
                                print("thickness status cont 1 ", thickness_container_1," cont 3 ",thickness_container_3,"cont 6 ", thickness_container_6)

                                for thickness_container_7 in range(0,8):
                                    
                                    for thickness_container_8 in range(0,8):
                                        
                                        for thickness_container_9 in range(0,8):
                                            
                                            for thickness_container_10 in range(0,8):
                                                    
                                                area = int(thickness_dict[thickness_container_1]*container_width + thickness_dict[thickness_container_2]*container_width + thickness_dict[thickness_container_3]*container_width + thickness_dict[thickness_container_4]*container_width + thickness_dict[thickness_container_5]*container_width + thickness_dict[thickness_container_6]*container_width + thickness_dict[thickness_container_7]*container_width + thickness_dict[thickness_container_8]*container_width + thickness_dict[thickness_container_9]*container_width + thickness_dict[thickness_container_10]*container_width )
                                                if area in range(area_ref -3 , area_ref +3):
                                                    chromosome_cont1 = np.repeat(thickness_dict[thickness_container_1],container_width/delta_x)
                                                    chromosome_cont2 = np.repeat(thickness_dict[thickness_container_2], container_width/delta_x)
                                                    chromosome_cont3 = np.repeat(thickness_dict[thickness_container_3], container_width/delta_x)
                                                    chromosome_cont4 = np.repeat(thickness_dict[thickness_container_4], container_width/delta_x)
                                                    chromosome_cont5 = np.repeat(thickness_dict[thickness_container_5], container_width/delta_x)
                                                    chromosome_cont6 = np.repeat(thickness_dict[thickness_container_6], container_width/delta_x)
                                                    chromosome_cont7 = np.repeat(thickness_dict[thickness_container_7], container_width/delta_x)
                                                    chromosome_cont8 = np.repeat(thickness_dict[thickness_container_8], container_width/delta_x)
                                                    chromosome_cont9 = np.repeat(thickness_dict[thickness_container_9], container_width/delta_x)
                                                    chromosome_cont10 = np.repeat(thickness_dict[thickness_container_10], container_width/delta_x)

                                                    chromosome_left = np.append(chromosome_cont1, [chromosome_cont2, chromosome_cont3, chromosome_cont4, chromosome_cont5, chromosome_cont6, chromosome_cont7, chromosome_cont8, chromosome_cont9, chromosome_cont10] )
                                                    chromosome_total = np.append(chromosome_left, np.flipud(chromosome_left))
                                                    chromosome = [ chromosome_total , area]
                                                    population.append(chromosome)
                                                    
                                                else:
                                                    continue
        
        population = np.array(population)
        index = list(range(1,len(population)+1))
        columns = {"Chromosome", "Area", "Fitness","Original Indi. No", "Cren Design", "Balance", "Lower Bound", "Upper Bound"}
        data = np.zeros((len(population),len(columns)), dtype = 'float')
        population_df = pd.DataFrame(data = data, index = index, columns = columns)
        
        population_df.Chromosome = population[:,0]
        population_df.Area = population[:,1]
        
        return population_df
    

    def brute_force_method_reference_7cont_8thick(self, bc):
        population = []
        t_min = bc.ix["minimum thickness"]
        t_ref = bc.ix["Reference thickness"]
        t_ref = 2.9
        delta_x = bc.ix["Stepsize horizontal"]
        half_width = bc.ix["Width"]/2
        number_of_containers = 8
        container_width = (half_width /2) / number_of_containers
        area_ref = int((half_width/2) * t_ref)
        
        thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}
            
        for thickness_container_1 in range(0,8):
            
            print(thickness_container_1)
                
            for thickness_container_2 in range(0,8):
                
                for thickness_container_3 in range(0,8):
                    
                    for thickness_container_4 in range(0,8):
                        
                        for thickness_container_5 in range(0,8):
                            
                            for thickness_container_6 in range(0,8):
                                
                                for thickness_container_7 in range(0,8):
                                    

                                    area = int(thickness_dict[thickness_container_1]*container_width + thickness_dict[thickness_container_2]*container_width + thickness_dict[thickness_container_3]*container_width + thickness_dict[thickness_container_4]*container_width + thickness_dict[thickness_container_5]*container_width + thickness_dict[thickness_container_6]*container_width + thickness_dict[thickness_container_7]*container_width )
                                    if area in range(area_ref -3 , area_ref +3):
                                        chromosome_cont1 = np.repeat(thickness_dict[thickness_container_1],container_width/delta_x)
                                        chromosome_cont2 = np.repeat(thickness_dict[thickness_container_2], container_width/delta_x)
                                        chromosome_cont3 = np.repeat(thickness_dict[thickness_container_3], container_width/delta_x)
                                        chromosome_cont4 = np.repeat(thickness_dict[thickness_container_4], container_width/delta_x)
                                        chromosome_cont5 = np.repeat(thickness_dict[thickness_container_5], container_width/delta_x)
                                        chromosome_cont6 = np.repeat(thickness_dict[thickness_container_6], container_width/delta_x)
                                        chromosome_cont7 = np.repeat(thickness_dict[thickness_container_7], container_width/delta_x)

                                        chromosome_left = np.append(chromosome_cont1, [chromosome_cont2, chromosome_cont3, chromosome_cont4, chromosome_cont5, chromosome_cont6, chromosome_cont7] )
                                        chromosome_total = np.append(chromosome_left, np.flipud(chromosome_left))
                                        chromosome = [ chromosome_total , area]
                                        population.append(chromosome)
                                        
                                    else:
                                        continue
        
        population = np.array(population)
        index = list(range(1,len(population)+1))
        columns = {"Chromosome", "Area", "Fitness","Original Indi. No", "Cren Design", "Balance", "Lower Bound", "Upper Bound"}
        data = np.zeros((len(population),len(columns)), dtype = 'float')
        population_df = pd.DataFrame(data = data, index = index, columns = columns)
        
        population_df.Chromosome = population[:,0]
        population_df.Area = population[:,1]
        
        return population_df
    



