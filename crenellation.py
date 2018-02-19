#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:48:08 2017

@author: Bart van der Lee
@project: MSc thesis 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import scipy.stats as ss
from scipy.stats import norm
import math
import multiprocessing as mp
from brute_force_method import brute_force_method
            

class CrenellationPattern:
    """
    A single solution, or in GA terms "an individual", which can have several attributes specific to the combinatorial optimization problem of crenellation design.
    
    Attributes:
        1. 
        2.
        3.
        4.
        5.
    
    """
    
    def __init__(self, ):

        self.


    def ConstructChromosomeSeed(self, SeedNumbers, delta_x, W, t_dict): #previous use_seed
        """
        Retrieves the shape of the seed design from the database based on the SeedNumber provided and scales the seed design shape 
        to fit the given boundary conditions (delta_x, W, t_dict).
        """
        t_pattern = CrenellationPattern.RetrieveSeedShape(self, SeedNumber)
            
        return t_pattern      
        
        
    def ConstructChromosomeRandom(self, delta_x, W, t_dict): #previous construct_chromosome
        """
        Construct chromosome with crenellation pattern based on boundary conditions given (delta_x, W, t_dict)
        """
        delta_a = bc.ix["crack step size"]
        a_max = bc.ix["Max crack length"]
        a_0 = bc.ix["Initial crack length"]
        total_a = a_max - a_0
        
        crenellation_type = str(bc.ix["Crenellation type"]) #determines the type of crenellation pattern that should be used
        cren_design = crenellation(bc,material)
        
        """
        Crennelation pattern is chosen according to the crenellation type
        """
        
        if crenellation_type == 'Random':  
            """
            Random crenellation pattern
            """
            t = cren_design.rand_thickness(bc,material)
            t = cren_design.apply_balance_init(t, bc, individual_no)

        elif crenellation_type == 'Uniform':
            """
            Uniform thickness plate
            """
            t = cren_design.uniform_thickness(bc,material)
        
        elif crenellation_type == 'Step thick':
            """
            Stepsize pattern
            """
            t = cren_design.step_thickness(bc,material)
        
        elif crenellation_type == 'Step sharp':
            """
            Sharp step pattern
            """
            t = cren_design.sharp_step(bc,material)
        
        elif crenellation_type == 'Reference':  
            """
            Reference study crenellation pattern
            """
            t = cren_design.ref_study_crenellation_simple(bc,material)
        
        elif crenellation_type == 'Lu 2015 coarse':  
            """
            Reference study crenellation pattern
            """
            t = cren_design.ref_study_cren_huber_5cont_8thick(bc,material)
            
        elif crenellation_type == 'Lu 2015 refined 10':  
            """
            Reference study crenellation pattern
            """
            t = cren_design.ref_study_cren_huber_10cont_8thick(bc,material)
                 
        elif crenellation_type == 'Lu 2015 refined 15':  
            """
            Reference study crenellation pattern
            """
            t = cren_design.ref_study_cren_huber_15cont_8thick(bc,material)
            
        t_pattern = t[0]
        return t_pattern
        
        
        
"""
#==============================================================================
#           Methods for creating crenellation designs 
#==============================================================================
"""        
        
        
        
        
        

    def ref_study_crenellation_huber_1(bc, material):
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        for i in range(0,cols):
            t[1,i] = i
            t[2,i] = i/100
            
        """
        Step sizes from reference study optimization
        """
        t[0][0:int(size/20)] = 1.9
        t[0][int((1*size)/20):int((2*size)/20)] = 2.03
        t[0][int((2*size)/20):int((3*size)/20)] = 2.18
        t[0][int((3*size)/20):int((4*size)/20)] = 2.48
        t[0][int((4*size)/20):int((5*size)/20)] = 2.78
        t[0][int((5*size)/20):int((6*size)/20)] = 2.93
        t[0][int((6*size)/20):int((7*size)/20)] = 3.53
        t[0][int((7*size)/20):int((8*size)/20)] = 3.53
        t[0][int((8*size)/20):int((9*size)/20)] = 3.82   
        t[0][int((9*size)/20):int((10*size)/20)] = 3.82
            

        t[0][int((10*size)/20):int((11*size)/20)] = 3.82
        t[0][int((11*size)/20):int((12*size)/20)] = 3.82
        t[0][int((12*size)/20):int((13*size)/20)] = 3.53
        t[0][int((13*size)/20):int((14*size)/20)] = 3.53
        t[0][int((14*size)/20):int((15*size)/20)] = 2.93
        t[0][int((15*size)/20):int((16*size)/20)] = 2.78
        t[0][int((16*size)/20):int((17*size)/20)] = 2.48
        t[0][int((17*size)/20):int((18*size)/20)] = 2.18   
        t[0][int((18*size)/20):int((19*size)/20)] = 2.03
        t[0][int((19*size)/20):int((20*size)/20)] = 1.9
        
        return t
            
    def ref_study_crenellation_uz_1(bc, material):
        """
        Crenellation pattern used to compare analytical fatigue model with experimental results from reference study by
        Uz et al. 
        """
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        for i in range(0,cols):
            t[1,i] = i
            t[2,i] = i/100
            
        total_container = 60 
            
        """
        Step sizes from reference study optimization in Uz et al.
        """
        t[0][0:int(size/total_container)] = 3.9
        t[0][int((1*size)/total_container):int((2*size)/total_container)] = 2.9
        t[0][int((2*size)/total_container):int((3*size)/total_container)] = 2.9
        t[0][int((3*size)/total_container):int((4*size)/total_container)] = 2.9
        t[0][int((4*size)/total_container):int((5*size)/total_container)] = 2.9
        t[0][int((5*size)/total_container):int((6*size)/total_container)] = 2.9
        t[0][int((6*size)/total_container):int((7*size)/total_container)] = 1.9
        t[0][int((7*size)/total_container):int((8*size)/total_container)] = 1.9
        t[0][int((8*size)/total_container):int((9*size)/total_container)] = 1.9   
        t[0][int((9*size)/total_container):int((10*size)/total_container)] = 1.9

        t[0][int((10*size)/total_container):int((11*size)/total_container)] = 1.9
        t[0][int((11*size)/total_container):int((12*size)/total_container)] = 4.15
        t[0][int((12*size)/total_container):int((13*size)/total_container)] = 4.15
        t[0][int((13*size)/total_container):int((14*size)/total_container)] = 4.15
        t[0][int((14*size)/total_container):int((15*size)/total_container)] = 4.15
        t[0][int((15*size)/total_container):int((16*size)/total_container)] = 4.15
        t[0][int((16*size)/total_container):int((17*size)/total_container)] = 4.15
        t[0][int((17*size)/total_container):int((18*size)/total_container)] = 4.15  
        t[0][int((18*size)/total_container):int((19*size)/total_container)] = 4.15
        t[0][int((19*size)/total_container):int((20*size)/total_container)] = 1.9
        
        t[0][int((20*size)/total_container):int((21*size)/total_container)] = 1.9
        t[0][int((21*size)/total_container):int((22*size)/total_container)] = 1.9
        t[0][int((22*size)/total_container):int((23*size)/total_container)] = 1.9
        t[0][int((23*size)/total_container):int((24*size)/total_container)] = 1.9
        t[0][int((24*size)/total_container):int((25*size)/total_container)] = 2.9
        t[0][int((25*size)/total_container):int((26*size)/total_container)] = 2.9
        t[0][int((26*size)/total_container):int((27*size)/total_container)] = 2.9
        t[0][int((27*size)/total_container):int((28*size)/total_container)] = 2.9   
        t[0][int((28*size)/total_container):int((29*size)/total_container)] = 2.9
        t[0][int((29*size)/total_container):int((30*size)/total_container)] = 15
        

        t[0][int((30*size)/total_container):int((31*size)/total_container)] = 2.9
        t[0][int((31*size)/total_container):int((32*size)/total_container)] = 2.9
        t[0][int((32*size)/total_container):int((33*size)/total_container)] = 2.9
        t[0][int((33*size)/total_container):int((34*size)/total_container)] = 2.9
        t[0][int((34*size)/total_container):int((35*size)/total_container)] = 2.9
        t[0][int((35*size)/total_container):int((36*size)/total_container)] = 2.9
        t[0][int((36*size)/total_container):int((37*size)/total_container)] = 1.9
        t[0][int((37*size)/total_container):int((38*size)/total_container)] = 1.9
        t[0][int((38*size)/total_container):int((39*size)/total_container)] = 1.9   
        t[0][int((39*size)/total_container):int((40*size)/total_container)] = 1.9

        t[0][int((40*size)/total_container):int((41*size)/total_container)] = 1.9
        t[0][int((41*size)/total_container):int((42*size)/total_container)] = 4.15
        t[0][int((42*size)/total_container):int((43*size)/total_container)] = 4.15
        t[0][int((43*size)/total_container):int((44*size)/total_container)] = 4.15
        t[0][int((44*size)/total_container):int((45*size)/total_container)] = 4.15
        t[0][int((45*size)/total_container):int((46*size)/total_container)] = 4.15
        t[0][int((46*size)/total_container):int((47*size)/total_container)] = 4.15
        t[0][int((47*size)/total_container):int((48*size)/total_container)] = 4.15  
        t[0][int((48*size)/total_container):int((49*size)/total_container)] = 4.15
        t[0][int((49*size)/total_container):int((50*size)/total_container)] = 1.9
        
        t[0][int((50*size)/total_container):int((51*size)/total_container)] = 1.9
        t[0][int((51*size)/total_container):int((52*size)/total_container)] = 1.9
        t[0][int((52*size)/total_container):int((53*size)/total_container)] = 1.9
        t[0][int((53*size)/total_container):int((54*size)/total_container)] = 1.9
        t[0][int((54*size)/total_container):int((55*size)/total_container)] = 2.9
        t[0][int((55*size)/total_container):int((56*size)/total_container)] = 2.9
        t[0][int((56*size)/total_container):int((57*size)/total_container)] = 2.9
        t[0][int((57*size)/total_container):int((58*size)/total_container)] = 2.9   
        t[0][int((58*size)/total_container):int((59*size)/total_container)] = 2.9
        t[0][int((59*size)/total_container):int((60*size)/total_container)] = 15

        
        return t
               
    def ref_study_uniform_uz_1( bc, material):
        """
        Uniform plate thickness used to compare analytical fatigue model with the experimental results from reference study
        by Uz et al.
        """
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        for i in range(0,cols):
            t[1,i] = i
            t[2,i] = i/100
            
        total_container = 60 
            
        """
        Step sizes from reference study optimization in Uz et al.
        """
        
        
        
        t[0][0:int(size/total_container)] = 2.9
        t[0][int((1*size)/total_container):int((2*size)/total_container)] = 2.9
        t[0][int((2*size)/total_container):int((3*size)/total_container)] = 2.9
        t[0][int((3*size)/total_container):int((4*size)/total_container)] = 2.9
        t[0][int((4*size)/total_container):int((5*size)/total_container)] = 2.9
        t[0][int((5*size)/total_container):int((6*size)/total_container)] = 2.9
        t[0][int((6*size)/total_container):int((7*size)/total_container)] = 2.9
        t[0][int((7*size)/total_container):int((8*size)/total_container)] = 2.9
        t[0][int((8*size)/total_container):int((9*size)/total_container)] = 2.9   
        t[0][int((9*size)/total_container):int((10*size)/total_container)] = 2.9

        t[0][int((10*size)/total_container):int((11*size)/total_container)] = 2.9
        t[0][int((11*size)/total_container):int((12*size)/total_container)] = 2.9
        t[0][int((12*size)/total_container):int((13*size)/total_container)] = 2.9
        t[0][int((13*size)/total_container):int((14*size)/total_container)] = 2.9
        t[0][int((14*size)/total_container):int((15*size)/total_container)] = 2.9
        t[0][int((15*size)/total_container):int((16*size)/total_container)] = 2.9
        t[0][int((16*size)/total_container):int((17*size)/total_container)] = 2.9
        t[0][int((17*size)/total_container):int((18*size)/total_container)] = 2.9  
        t[0][int((18*size)/total_container):int((19*size)/total_container)] = 2.9
        t[0][int((19*size)/total_container):int((20*size)/total_container)] = 2.9
        
        t[0][int((20*size)/total_container):int((21*size)/total_container)] = 2.9
        t[0][int((21*size)/total_container):int((22*size)/total_container)] = 2.9
        t[0][int((22*size)/total_container):int((23*size)/total_container)] = 2.9
        t[0][int((23*size)/total_container):int((24*size)/total_container)] = 2.9
        t[0][int((24*size)/total_container):int((25*size)/total_container)] = 2.9
        t[0][int((25*size)/total_container):int((26*size)/total_container)] = 2.9
        t[0][int((26*size)/total_container):int((27*size)/total_container)] = 2.9
        t[0][int((27*size)/total_container):int((28*size)/total_container)] = 2.9   
        t[0][int((28*size)/total_container):int((29*size)/total_container)] = 2.9
        t[0][int((29*size)/total_container):int((30*size)/total_container)] = 15
        

        t[0][int((30*size)/total_container):int((31*size)/total_container)] = 2.9
        t[0][int((31*size)/total_container):int((32*size)/total_container)] = 2.9
        t[0][int((32*size)/total_container):int((33*size)/total_container)] = 2.9
        t[0][int((33*size)/total_container):int((34*size)/total_container)] = 2.9
        t[0][int((34*size)/total_container):int((35*size)/total_container)] = 2.9
        t[0][int((35*size)/total_container):int((36*size)/total_container)] = 2.9
        t[0][int((36*size)/total_container):int((37*size)/total_container)] = 2.9
        t[0][int((37*size)/total_container):int((38*size)/total_container)] = 2.9
        t[0][int((38*size)/total_container):int((39*size)/total_container)] = 2.9   
        t[0][int((39*size)/total_container):int((40*size)/total_container)] = 2.9

        t[0][int((40*size)/total_container):int((41*size)/total_container)] = 2.9
        t[0][int((41*size)/total_container):int((42*size)/total_container)] = 2.9
        t[0][int((42*size)/total_container):int((43*size)/total_container)] = 2.9
        t[0][int((43*size)/total_container):int((44*size)/total_container)] = 2.9
        t[0][int((44*size)/total_container):int((45*size)/total_container)] = 2.9
        t[0][int((45*size)/total_container):int((46*size)/total_container)] = 2.9
        t[0][int((46*size)/total_container):int((47*size)/total_container)] = 2.9
        t[0][int((47*size)/total_container):int((48*size)/total_container)] = 2.9  
        t[0][int((48*size)/total_container):int((49*size)/total_container)] = 2.9
        t[0][int((49*size)/total_container):int((50*size)/total_container)] = 2.9
        
        t[0][int((50*size)/total_container):int((51*size)/total_container)] = 2.9
        t[0][int((51*size)/total_container):int((52*size)/total_container)] = 2.9
        t[0][int((52*size)/total_container):int((53*size)/total_container)] = 2.9
        t[0][int((53*size)/total_container):int((54*size)/total_container)] = 2.9
        t[0][int((54*size)/total_container):int((55*size)/total_container)] = 2.9
        t[0][int((55*size)/total_container):int((56*size)/total_container)] = 2.9
        t[0][int((56*size)/total_container):int((57*size)/total_container)] = 2.9
        t[0][int((57*size)/total_container):int((58*size)/total_container)] = 2.9   
        t[0][int((58*size)/total_container):int((59*size)/total_container)] = 2.9
        t[0][int((59*size)/total_container):int((60*size)/total_container)] = 15

        
        return t
        
    def uniform_thickness( bc, material):
        """
        Method to construct a plate of uniform thickness given the boundary conditions
        """
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        t_ref = bc.ix["Reference thickness"]
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        t[0,:] = t_ref
        for i in range(0,cols):
            t[1,i] = i
            t[2,i] = i/100
        
        return t
        
    def small_variations(self,bc,material):
        """
        Method to construct a plate with a crenellation pattern with small variations
        """
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        
        for i in range(0,cols-10,20):
            t[0,i:i+10] = 1.75
            
            for k in range(10,cols,20):
                t[0,k:k+10] = 2.25
        
        return t[0]
        
    def large_variations(self,bc,material):
        """
        Method to construct a plate with a crenellation pattern with larger variations
        """
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        
        for i in range(0,cols-5,10):
            t[0,i:i+5] = 2.50
            
            for k in range(5,cols,10):
                t[0,k:k+5] = 1.50
        
            
        return t[0] 
        
    def step_thickness(self, bc, material):
        """
        Method to construct plate with 1 gradual step change in the middle of the plate
        """
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        for i in range(0,cols):
            t[1,i] = i
            t[2,i] = i/100

        """
        Step sizes with large middle
        """
        t[0][0:int(size/9)] = 2.25
        t[0][int((1*size)/9):int((2*size)/9)] = 1.25
        t[0][int((2*size)/9):int((3*size)/9)] = 1.75
        t[0][int((3*size)/9):int((4*size)/9)] = 2.25
        t[0][int((4*size)/9):int((5*size)/9)] = 2.5
        t[0][int((5*size)/9):int((6*size)/9)] = 2.25
        t[0][int((6*size)/9):int((7*size)/9)] = 1.75
        t[0][int((7*size)/9):int((8*size)/9)] = 1.25
        t[0][int((8*size)/9):int((9*size)/9)] = 2.25
        
        return t[0]

    def sharp_step(self, bc, material):
        """
        profile with 1 sharp step change in the middle of the plate
        """
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        size = np.int(half_width/delta_x)
        t = np.zeros((4,size), dtype = 'float')
        cols = t.shape[1]
        for i in range(0,cols):
            t[1,i] = i
            t[2,i] = i/100

        """
        Step sizes with large middle
        """
        t[0][0:int((13*size)/27)] = 1.5
        t[0][int((13*size)/27):int((14*size)/27)] = 14.66
        t[0][int((14*size)/27):int((27*size)/27)] = 1.5

        return t[0]
        
    def ref_study_cren_huber_5cont_8thick_brute(self, bc, material):
        """
        Method to randomly pick a crenellation pattern from the solutions space given by the reference study by 
        Huber et al. Plates have 5 containers with 8 thicknesses.
        """
                
        population_eval_sorted_5cont_8thick = pd.read_pickle("/Users/Bart/Google Drive/6. Education/Master/4. MSc Thesis/Python code/classes/version2/reference_Lu_brute_force_5cont_8thick")

        
        possible_individuals = population_eval_sorted_5cont_8thick.loc[population_eval_sorted_5cont_8thick["Area"] == 219]
        chosen = np.random.choice(possible_individuals.index,1)
        t = [possible_individuals.Chromosome[chosen[0]],0]
        
        return t
        
    def ref_study_cren_huber_5cont_8thick(self, bc, material):
        """
        Method to randomly construct a crenellation pattern according to the reference research by Huber et al.
        5 containers with each 8 possible thicknesses
        """
        t_ref = bc.ix["Reference thickness"]
        half_width = bc.ix["Width"]/4
        area_ref = t_ref *( half_width*2)
        number_of_containers = bc.ix["number_of_containers"]
        container_width = (half_width) / number_of_containers
        delta_x = bc.ix["Stepsize horizontal"]
        area_chromosome = []
        
        while area_chromosome == []:
            thickness_options = [0,1,2,3,4,5,6,7]
            thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}
            x1 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x2 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x3 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x4 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x5 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)


            chromosome_left = np.append([x1], [x2, x3, x4, x5])
            chromosome = np.append(chromosome_left,np.flipud(chromosome_left))
            area_calculated = np.sum(chromosome)
            print(chromosome)
            print(area_calculated)
            print(area_ref)
            
            if  area_ref - 5 < area_calculated < area_ref +5:
                area_chromosome = area_calculated
                t = [chromosome,0]

        return t
        
        
    def ref_study_cren_huber_10cont_8thick(self, bc, material):
        """
        Method to randomly construct a crenellation pattern according to the reference research by Huber et al.
        10 containers with each 8 possible thicknesses
        """
        t_ref = bc.ix["Reference thickness"]
        half_width = bc.ix["Width"]/4
        area_ref = t_ref *( half_width*2)
        number_of_containers = bc.ix["number_of_containers"]
        container_width = half_width / number_of_containers
        delta_x = bc.ix["Stepsize horizontal"]
        area_chromosome = []
        
        while area_chromosome == []:
            thickness_options = [0,1,2,3,4,5,6,7]
            thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}
            x1 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x2 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x3 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x4 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x5 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x6 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x7 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x8 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x9 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x10 =np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)

            chromosome_left = np.append([x1], [x2, x3, x4, x5, x6, x7, x8, x9, x10])
            chromosome = np.append(chromosome_left,np.flipud(chromosome_left))
            area_calculated = np.sum(chromosome)
            print(chromosome)
            print(area_calculated)
            
            if  area_ref - 2 < area_calculated < area_ref +2:
                area_chromosome = area_calculated
                t = [chromosome,0]

        return t
        
    def ref_study_cren_huber_15cont_8thick(self, bc, material):
        """
        Method to randomly construct a crenellation pattern according to the reference research by Huber et al.
        15 containers with each 8 possible thicknesses
        """ 
        t_ref = bc.ix["Reference thickness"]
        half_width = bc.ix["Width"]/4
        area_ref = t_ref *( half_width*2)
        number_of_containers = bc.ix["number_of_containers"]
        container_width = half_width / number_of_containers
        delta_x = bc.ix["Stepsize horizontal"]
        area_chromosome = []
        
        while area_chromosome == []:
            thickness_options = [0,1,2,3,4,5,6,7]
            thickness_dict = {0: 1.9 ,1: 2.22 , 2: 2.54, 3: 2.86, 4: 3.19, 5: 3.51, 6: 3.83, 7:4.15}
            x1 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x2 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x3 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x4 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x5 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x6 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x7 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x8 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x9 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x10 =np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x11 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x12 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x13 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x14 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)
            x15 = np.repeat(thickness_dict[np.random.choice(thickness_options)],container_width/delta_x)

            chromosome_left = np.append([x1], [x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15])
            chromosome = np.append(chromosome_left,np.flipud(chromosome_left))
            area_calculated = np.sum(chromosome)
            print(chromosome)
            print(area_calculated)
            
            if  area_ref - 2 < area_calculated < area_ref +2:
                area_chromosome = area_calculated
                t = [chromosome,0]

        return t
        
        
        
    """        
    #==============================================================================
    #         Crenellation area methods
    #==============================================================================
    """        
        
    def cal_cren_area(self, thickness_pattern, cren_design, bc):

        delta_x = bc.ix["Stepsize horizontal"]           
        thickness_pattern['Width'][0] = thickness_pattern['Width'][2]
        thickness_pattern['Width'][1] = thickness_pattern['Width'][2]

        """
        Calculate the areas per container based on differnt crack length a
        """
        a = np.array(cren_design['a'])
        x = np.array(thickness_pattern.ix[:,0])

        area_cren = crenellation.calculate_cren_area
        
#        print("start evaluating integral")
        X = np.fromfunction(lambda i,j: area_cren(a[i],x[j],delta_x), (len(a),len(x)), dtype = 'int')
#        print("area calculation done")
        
        thickness_pattern = thickness_pattern["thickness"]
        thickness_pattern = np.array(thickness_pattern)
        thickness_pattern_trans = np.transpose(thickness_pattern)      

        area_cren = X * thickness_pattern_trans
        area_cren = np.around(area_cren,decimals = 3)
        
        cren_design["area"] = np.sum(area_cren, axis=1) 
        
        return cren_design
        
    """        
    #==============================================================================
    #         
    #==============================================================================
    """        
        
    def rand_thickness(self, bc,material):
        
        delta_x = bc.ix["Stepsize horizontal"]     #mm
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                   #mm
        delta_t_min =  bc.ix["Layer thickness"]    #mm
        t_ref = bc.ix["Reference thickness"]       #mm
        t_0 = bc.ix["minimum thickness"]           #mm
        t_max = bc.ix["maximum thickness"]         #mm
        t_bound = 2*delta_t_min
        rounding = int(delta_t_min * 10)
        dimension = np.int(half_width/delta_x)
        t = np.zeros((4,dimension), dtype = 'float')
        cols = t.shape[1]
        t_start = t_ref

        #delta_a_max = bc.ix[""]  #mm
        """
        should become the decoding of chromosome, instead of random thickness distribution
        Now: random thickness profile such that total surface of crenellation equals total surface
        of the reference panel (equal weight constraint)
        """
        
        for i in range(0, cols):
            
            if i == 0:
                t[0,i] = t_start
                t[1,i] = 1
                t[2,i] = t_start-t_ref
                 
            elif i-1 > 0.5*cols:
                t[0,i] = t[0,int(0.5*cols)-(i-int(0.5*cols))]        #choosing thickness of next column symmetrically
                t[1,i] = t[1,i-1]+1 #index of next column
                t[2,i] = t[0,i]-t_ref #contribution to the balance with respect to the reference panel
                t[3,i] = i #index of current column in whole numbers
        
            elif t[0,i-1] < t_0 + t_bound:        #if previous t-value is within range of lower bound t_0
                t[0,i] = np.round(np.random.uniform(t_0,t[0,i-1]+t_bound), rounding)#choosing thickness of next column
                t[1,i] = t[1,i-1]+1 #index of next column
                t[2,i] = (t[0,i]-t_ref) #contribution to the balance with respect to the reference panel
                t[3,i] = i #index of current column in whole numbers
        
            elif t[0,i-1]> t_max-t_bound:     #if previous t-value is within range of upper bound t_max
                t[0,i] = np.round(np.random.uniform(t[0,i-1]-t_bound,t_max), rounding)#choosing thickness of next column
                t[1,i] = t[1,i-1]+1 #index of next column
                t[2,i] = (t[0,i]-t_ref)#contribution to the balance with respect to the reference panel
                t[3,i] = i #index of current column in whole numbers
            
            else:                             #when previous t-value is not within range of the either thickness bounds
                t[0,i] = np.round(np.random.uniform(t[0,i-1]-t_bound,t[0,i-1]+t_bound), rounding)#choosing thickness of next column
                t[1,i] = t[1,i-1]+1 #index of next column
                t[2,i] = (t[0,i]-t_ref)#contribution to the balance with respect to the reference panel
                t[3,i] = i #index of current column in whole numbers
        
#        print("lower than t_min, init",np.count_nonzero(t[0]<1))
#        print("higher than than t_max, init",np.count_nonzero(t[0]>3))

        return t


        
    """      
    #==============================================================================
    #         Balancing methods
    #==============================================================================
    """     
        
    def apply_balance_init(self, t_array,bc,individual_no):
        global new_balance
        cols = t_array.shape[1]
        t_ref = bc.ix["Reference thickness"]  #mm
        t_max = bc.ix["maximum thickness"]
        t_min = bc.ix["minimum thickness"]
        
        current_balance = np.sum(t_array[2,:])
        balance_difference = current_balance #spreading excess or shortage of surface compared to Aref  
                          
        if balance_difference < 0:
            t_change = (abs(balance_difference) / cols) 
            t_array[0,:] = t_array[0,:] + t_change
            t_array[2,:] = t_array[0,:] - t_ref
            new_balance = np.sum(t_array[2,:])
            t = t_array[0]
        
            
        elif balance_difference > 0:
            t_change = (abs(balance_difference) / cols) 
            t_array[0,:] = t_array[0,:] - t_change
            t_array[2,:] = t_array[0,:] - t_ref
            new_balance = np.sum(t_array[2,:])
            t = t_array[0]

            
        #determin the upper and lower bounds
        t_upper_bound = t - t_max
        t_lower_bound = (t - t_min) * -1
        # -1 to make the lower bound shortage number positive floats 
        """
        Spreading upper bound excess thickness over the left and right sides of the current container
        """
        t = crenellation.respread_upper_bound(self, t_upper_bound, t, individual_no)        
            
        """
        Taking lower bound shortage of thickness from the left and right sides of the current container to fill the container to t_min
        """
        t = crenellation.respread_lower_bound(self, t_lower_bound, t, individual_no)
        
        t_array[0] = t
#        print("lower than tmin",np.count_nonzero(t_array[0]<t_min))
#        print("higher than tmax",np.count_nonzero(t_array[0]>t_max))
        return t_array
#        print("The end balance is:", new_balance, "after redistributing", current_balance)
        
    def respread_upper_bound(self, t_upper_bound, t, individual_no):
        for c in range(0,len(t_upper_bound)):
            
            if t_upper_bound[c] > 0:
#                print('current ',c, "individual number ",individual_no)
                # The upper bound shortage is acquired equally from both sides
                left_shortage = t_upper_bound[c] / 2
                right_shortage = t_upper_bound[c] / 2
                
                for p in range(0,2):
                    # If you are in the second run and there is still an excess on one of the side
                    # that side's boundary has been reached, so this excess should be spread over the
                    # opposite side.
                    if p == 1 and left_shortage > 0:
                        right_shortage += left_shortage
                        left_shortage = 0
                    if p == 1 and right_shortage > 0:
                        left_shortage += right_shortage
                        right_shortage = 0
                    
                    if left_shortage > 0 or right_shortage > 0:
                        # If there is any shortage, start looking for excess on sides
                        for s in range(1, max(c-1,len(t)-c-1)):
                            if c-s >= 0:
                                
                                left_excess = t_upper_bound[c-s]
                                # take shortage over left side
                                """
                                verbeterd 
                                """
                                if left_excess < 0 and left_shortage > 0 and left_shortage > abs(left_excess):
#                                    print("1. Taking from left ", left_excess, "to        ", t[c-s], t[c], "at container ",c-s)
                                    t_transfer = abs(left_excess) #take as much as possible from the excess needed for the shortage
                                    t[c-s] = t[c-s]+ t_transfer
                                    t[c] = t[c] - t_transfer
                                    t_upper_bound[c-s] = t_upper_bound[c-s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_upper_bound[c] = t_upper_bound[c] - t_transfer
                                    left_shortage = left_shortage - t_transfer
#                                    print("1. Took from left ", t_transfer,    "such that ", t[c-s], t[c], "at container ",c-s)
                                elif left_excess < 0 and left_shortage > 0 and left_shortage < abs(left_excess):
#                                    print("2. Taking from left ", left_excess, "to        ", t[c-s], t[c], "at container ",c-s)
                                    t_transfer = left_shortage #take as much as possible from the excess needed for the shortage
                                    t[c-s] = t[c-s]+ t_transfer
                                    t[c] = t[c] - t_transfer
                                    t_upper_bound[c-s] = t_upper_bound[c-s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_upper_bound[c] = t_upper_bound[c] - t_transfer
                                    left_shortage = 0
#                                    print("2. Took from left ", t_transfer,    "such that ", t[c-s], t[c], "at container ",c-s)
                                                                                                        
                            if c+s < len(t_upper_bound):
                                right_excess = t_upper_bound[c+s]
                                if right_excess < 0 and right_shortage > 0 and right_shortage > abs(right_excess):
#                                    print("3. Taking from right ", right_excess,"to         ", t[c+s], t[c], "at container ",c+s)
                                    t_transfer = abs(right_excess) #take as much as possible from the excess needed for the shortage
                                    # Shortage exceeds the excess. So there will remain some shortage
                                    t[c+s] = t[c+s] + t_transfer
                                    t[c] = t[c] - t_transfer
                                    t_upper_bound[c+s] = t_upper_bound[c+s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_upper_bound[c] = t_upper_bound[c] - t_transfer
                                    right_shortage = right_shortage - t_transfer
#                                    print("3. Took from right ", t_transfer,   "such that   ", t[c+s], t[c], "at container ",c+s)
                                    
                                elif right_excess < 0 and right_shortage > 0 and right_shortage < abs(right_excess):
 #                                   print("4. Taking from right ", right_excess,"to          ", t[c+s], t[c], "at container ",c+s)
                                    t_transfer = right_shortage #take as much as possible from the excess needed for the shortage
                                    # Enough excess to fill the shortage. So no shortage remains
                                    t[c+s] = t[c+s] + t_transfer
                                    t[c] = t[c] - t_transfer
                                    t_upper_bound[c+s] = t_upper_bound[c+s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_upper_bound[c] = t_upper_bound[c] - t_transfer
                                    right_shortage = 0
#                                    print("4. Took from right ", t_transfer,   "such that   ",t[c+s], t[c], "at container ",c+s)
                                    
                            if left_shortage < 0 and right_shortage < 0:
                                break
                            
                            """
                            if the right or left boundary is reached and there is still a shortage, go to second run p = 1 to continue on other side
                            """
                            # right boundary reached, 
                            #if c+s >=0 and left_shortage <= 0 and right_shortage >= 0:
                             #   break
                            # left boundary reached,
                            #if c-s <=0 and right_shortage <= 0 and left_shortage >= 0:
                             #   break  
        
        return t
                            
          
    def respread_lower_bound(self, t_lower_bound, t, individual_no):
        for c in range(0,len(t_lower_bound)):
            
            if t_lower_bound[c] > 0:
#                print('current ',c, "individual number ",individual_no)
                # The lower bound shortage is acquired equally from both sides
                left_shortage = t_lower_bound[c] / 2
                right_shortage = t_lower_bound[c] / 2
                
                for p in range(0,2):
                    # If you are in the second run and there is still an excess on one of the side
                    # that side's boundary has been reached, so this excess should be spread over the
                    # opposite side.
                    if p == 1 and left_shortage > 0:
                        right_shortage += left_shortage
                        left_shortage = 0
                    if p == 1 and right_shortage > 0:
                        left_shortage += right_shortage
                        right_shortage = 0
                    
                    if left_shortage > 0 or right_shortage > 0:
                        # If there is any shortage, start looking for excess on sides
                        for s in range(1, max(c-1,len(t)-c-1)):
                            if c-s >= 0:
                                
                                left_excess = t_lower_bound[c-s]
                                # take shortage over left side
                                """
                                verbeterd 
                                """
                                if left_excess < 0 and left_shortage > 0 and left_shortage > abs(left_excess):
#                                    print("1. Taking from left ", left_excess, "to        ", t[c-s], t[c], "at container ",c-s)
                                    t_transfer = abs(left_excess) #take as much as possible from the excess needed for the shortage
                                    t[c-s] = t[c-s]- t_transfer
                                    t[c] = t[c] + t_transfer
                                    t_lower_bound[c-s] = t_lower_bound[c-s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_lower_bound[c] = t_lower_bound[c] - t_transfer
                                    left_shortage = left_shortage - t_transfer
#                                    print("1. Took from left ", t_transfer,    "such that ", t[c-s], t[c], "at container ",c-s)
                                elif left_excess < 0 and left_shortage > 0 and left_shortage < abs(left_excess):
#                                   print("2. Taking from left ", left_excess, "to        ", t[c-s], t[c], "at container ",c-s)
                                    t_transfer = left_shortage #take as much as possible from the excess needed for the shortage
                                    t[c-s] = t[c-s]- t_transfer
                                    t[c] = t[c] + t_transfer
                                    t_lower_bound[c-s] = t_lower_bound[c-s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_lower_bound[c] = t_lower_bound[c] - t_transfer
                                    left_shortage = 0
#                                   print("2. Took from left ", t_transfer,    "such that ", t[c-s], t[c], "at container ",c-s)
                                                                                                        
                            if c+s < len(t_lower_bound):
                                right_excess = t_lower_bound[c+s]
                                if right_excess < 0 and right_shortage > 0 and right_shortage > abs(right_excess):
#                                    print("3. Taking from right ", right_excess,"to         ", t[c+s], t[c], "at container ",c+s)
                                    t_transfer = abs(right_excess) #take as much as possible from the excess needed for the shortage
                                    # Shortage exceeds the excess. So there will remain some shortage
                                    t[c+s] = t[c+s] - t_transfer
                                    t[c] = t[c] + t_transfer
                                    t_lower_bound[c+s] = t_lower_bound[c+s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_lower_bound[c] = t_lower_bound[c] - t_transfer
                                    right_shortage = right_shortage - t_transfer
#                                   print("3. Took from right ", t_transfer,   "such that   ", t[c+s], t[c], "at container ",c+s)
                                    
                                elif right_excess < 0 and right_shortage > 0 and right_shortage < abs(right_excess):
#                                    print("4. Taking from right ", right_excess,"to          ", t[c+s], t[c], "at container ",c+s)
                                    t_transfer = right_shortage #take as much as possible from the excess needed for the shortage
                                    # Enough excess to fill the shortage. So no shortage remains
                                    t[c+s] = t[c+s] - t_transfer
                                    t[c] = t[c] + t_transfer
                                    t_lower_bound[c+s] = t_lower_bound[c+s] + t_transfer #adjust the balance for thickness in the t_bound array
                                    t_lower_bound[c] = t_lower_bound[c] - t_transfer
                                    right_shortage = 0
#                                    print("4. Took from right ", t_transfer,   "such that   ",t[c+s], t[c], "at container ",c+s)
                                    
                            if left_shortage < 0 and right_shortage < 0:
                                break
                            
                            """
                            if the right or left boundary is reached and there is still a shortage, go to second run p = 1 to continue on other side
                            """
                            # right boundary reached, 
                            #if c+s >=0 and left_shortage <= 0 and right_shortage >= 0:
                             #   break
                            # left boundary reached,
                            #if c-s <=0 and right_shortage <= 0 and left_shortage >= 0:
                             #   break
        return t

            
    def apply_balance_crossover(self, t,bc, individual_no):
        t_ref = bc.ix["Reference thickness"]       #mm
        current_balance = np.sum(t - t_ref)
        t_max = bc.ix["maximum thickness"]
        t_min = bc.ix["minimum thickness"]
        cols = t.shape[0]
        
        if current_balance < 0:
            t_change = (abs(current_balance) / cols)
            t = t + t_change
            new_balance = np.sum(t-t_ref)

        
        elif current_balance > 0:
            t_change = (abs(current_balance) / cols)
            t = t - t_change
            new_balance = np.sum(t-t_ref)

            
        t_upper_bound = t - t_max 
        t_lower_bound = (t - t_min) * -1
        # -1 to make the lower bound shortage number positive floats  
        # split residue in neighbouring containers
        """
        Spreading upper bound excess thickness over the left and right sides of the current container
        """
        t = crenellation.respread_upper_bound(self, t_upper_bound, t, individual_no)
                 
        """
        Taking lower bound shortage of thickness from the left and right sides of the current container to fill the container to t_min
        """
        t = crenellation.respread_lower_bound(self, t_lower_bound, t, individual_no)
        return t
           
        
    def apply_balance_mutation(self, t, bandwidth_left, bandwidth_right, current_balance, mutated_balance, individual_chromosome, bc):
        t_ref = bc.ix["Reference thickness"]       #mm
        cols_middle = t.shape[1]-2 #columns in between the first and last container
        difference = abs(current_balance - mutated_balance)
        t_change = difference / cols_middle
        
        if current_balance < 0 and mutated_balance < 0 or current_balance > 0 and mutated_balance > 0:
            direction = mutated_balance - current_balance
            
            if direction > 0: #if mutated_balance is larger than current_balance, t_change must be subtracted
                t[0,0:cols_middle] = t[0,0:cols_middle] - t_change
            
            elif direction < 0: #if mutated_balance is smaller than current_balance, t_change must be added
                t[0,0:cols_middle] = t[0,0:cols_middle] + t_change

        elif current_balance < 0: #current_balance is lower than the mutated_balance, t_change must be subtracted
            t[0,0:cols_middle] = t[0,0:cols_middle] - t_change

        elif mutated_balance < 0:  #mutated balance is lower than the current_balance, t_change must be added
            t[0,0:cols_middle] = t[0,0:cols_middle] + t_change


#        if current_balance < 0 and mutated_balance < 0 or current_balance > 0 and mutated_balance > 0:
#            direction = mutated_balance - current_balance
#            
#            if direction > 0: #if mutated_balance is larger than current_balance, t_change must be subtracted
#            
#                for j in range(1,cols_middle):
#                    t_change = min(t_bound - abs(t[0,j-1]-t[0,j]), t_change) 
#                    t_balance_old = t[0,j] - t_ref
#                    t[0,j] = t[0,j] - t_change
#                    t_balance_new = t[0,j] - t_ref
#                    mutated_balance = mutated_balance - abs(t_balance_old-t_balance_new)
#                    difference = abs(current_balance - mutated_balance)
#                    t_change = difference / (cols_middle - j)
#                    
#            elif direction < 0: #if mutated_balance is smaller than current_balance, t_change must be added
#                
#                for j in range(1,cols_middle):
#                    t_change = min(t_bound - abs(t[0,j-1]-t[0,j]), t_change) 
#                    t_balance_old = t[0,j] - t_ref
#                    t[0,j] = t[0,j] + t_change
#                    t_balance_new = t[0,j] - t_ref
#                    mutated_balance = mutated_balance - abs(t_balance_old-t_balance_new)
#                    difference = abs(current_balance - mutated_balance)
#                    t_change = difference / (cols_middle - j)
#                    
#        elif mutated_balance < 0:  #mutated balance is lower than the current_balance, t_change must be added
#            
#            for j in range(1,cols_middle):
#                t_change = min(t_bound - abs(t[0,j-1]-t[0,j]), t_change) 
#                t_balance_old = t[0,j] - t_ref
#                t[0,j] = t[0,j] + t_change
#                t_balance_new = t[0,j] - t_ref
#                mutated_balance = mutated_balance - abs(t_balance_old-t_balance_new)
#                difference = abs(current_balance - mutated_balance)
#                t_change = difference / (cols_middle - j)
#                   
#        elif current_balance < 0: #current_balance is lower than the mutated_balance, t_change must be subtracted
#            
#            for j in range(1,cols_middle):
#                t_change = min(t_bound - (t[0,j-1]-t[0,j]), t_change) 
#                t_balance_old = t[0,j] - t_ref
#                t[0,j] = t[0,j] - t_change
#                t_balance_new = t[0,j] - t_ref
#                mutated_balance = mutated_balance - abs(t_balance_old-t_balance_new)
#                difference = abs(current_balance - mutated_balance)
#                t_change = difference / (cols_middle - j)
            
        mutated_balance = np.sum(t-t_ref)
        
        individual_chromosome[bandwidth_left:bandwidth_right+1] = t
        
        return individual_chromosome
            
    def create_dataframe(self, lifetime, thickness_pattern, bc, material):
        W = int(bc.ix["Width"])                    #mm
        half_width = np.int(W/2)                     #mm
        delta_x = bc.ix["Stepsize horizontal"]
        a_0 = bc.ix["Initial crack length"]
        a_max = bc.ix["Max crack length"]
        delta_a = bc.ix["crack step size"]
        total_a = a_max - a_0
        size = int(total_a / delta_a)
        index = range(0,size)
        thickness = pd.DataFrame(data=np.zeros((size,1)),index=index, columns={'thickness'}, dtype='float') 

        index= range(0,np.int(half_width/delta_x))
        array_thickness = thickness_pattern[:].round(decimals=2)
        thickness_graph = pd.DataFrame(data = array_thickness, index=index,columns={'thickness'}, dtype='float')
        
        """
        calculates the area of the crenellation pattern from each point until the width of the panel
        """
        
        array_width = np.linspace(0,half_width, num=len(index)).round(decimals=2)
        half_width_graph = pd.DataFrame(data=array_width , index=index, columns={'Width'}, dtype='float')
        half_width_graph = half_width_graph.join(thickness_graph)
        a_loc = int(a_0/delta_a)

#        print("creating crenellation design dataframe")
        for i in range(len(thickness)):
            thickness['thickness'][i] = thickness_pattern[i+(a_loc)] #thickness start at the tip of the initial crack
 
        lifetime = lifetime.join(thickness)

        a = [lifetime, half_width_graph]
        
        return a

        
#==============================================================================
#      Prior solution   
#==============================================================================
        
#        for i in range(0,len(cren_design)):
#            a = cren_design["a"][i]
#        
#            """
#            Integration formula 5
#            """
#            thickness_pattern['A/X'] = ((a/thickness_pattern['Width'])**2)
#            thickness_pattern['A/X-1'] = ((a/(thickness_pattern['Width']-delta_x))**2)
#            thickness_pattern.ix[thickness_pattern['A/X'] > 1, 'A/X'] = 0
#            thickness_pattern.ix[thickness_pattern['A/X-1'] > 1, 'A/X-1'] = 0  
#            
#            thickness_pattern['Area Cren'] = thickness_pattern['thickness']*thickness_pattern['Width']* np.sqrt(1 - thickness_pattern['A/X'])   - (thickness_pattern['thickness']*(thickness_pattern['Width']-delta_x) * np.sqrt(1 - thickness_pattern['A/X-1']))
#            thickness_pattern.ix[thickness_pattern['A/X'] == 0, 'Area Cren'] = 0
#
#
#            total_area_cren = sum(thickness_pattern['Area Cren'])
#            cren_design["area"][i] = total_area_cren
#            print(i, total_area_cren) 
#            """
#            Sum all the area contributions
#            """


    def calculate_cren_area(a,x, delta_x):
        """
        Integral evaluation
        """

#        pool = mp.Pool(processes=4)
#        results = [pool.apply(crenellation.divide(a,x)) for a in a]
#    
#        
#        print("calculating A")
        A = 1 - (np.true_divide(a,x)**2)
        k = x - delta_x 
#        print("calculating A_1")
        A_1 = 1 - (np.true_divide(a,k)**2)
        
#        print("removing negative values")
        negative_values = A < 0
        A[negative_values] = 0

        negative_values_2 = A_1 < 0
        A_1[negative_values_2] = 0
        
#        print("calculating square roots")
        area_cren = (x * np.sqrt(A))   - ((k) * np.sqrt(A_1))

        return area_cren
        
#==============================================================================
#         Mutation crenellation pattern        
#==============================================================================
        
        
        
    def rand_thickness_mutation(self, individual_chromosome, bandwidth_left, bandwidth_right, thickness_left, thickness_right, bc):
        delta_t_min =  bc.ix["Layer thickness"]    #mm
        t_0 = bc.ix["minimum thickness"]           #mm
        t_max = bc.ix["maximum thickness"]         #mm
        t_bound = 1*delta_t_min
        t_ref = bc.ix["Reference thickness"]
        
        cols = (bandwidth_right - bandwidth_left)+1
        t = np.zeros((1,cols), dtype = 'float')
        t_start = thickness_left
        t_end = thickness_right
        t[0,0] = t_start
        t[0,cols-1] = t_end
        
        for i in range(1, cols-1):
            
            y = t_bound*((cols-1)-i)
            difference = t[0,i-1]-t_end
            margin = y - abs(difference)
            
            if difference > 0 and margin <= 0:
                t[0,i] = t[0,i-1] - t_bound
            
            elif difference < 0 and margin <= 0:
                t[0,i] = t[0,i-1] + t_bound
                
            elif t[0,i-1] < t_0 + t_bound:        #if previous t-value is within range of lower bound t_0
                t[0,i] = np.random.uniform(t_0,t[0,i-1]+t_bound)#choosing thickness of next column

            elif t[0,i-1] > t_max-t_bound:     #if previous t-value is within range of upper bound t_max
                t[0,i] = np.random.uniform(t[0,i-1]-t_bound,t_max)#choosing thickness of next column
          
            else:                             #when previous t-value is not within range of the either thickness bounds
                t[0,i] = np.random.uniform(t[0,i-1]-t_bound,t[0,i-1]+t_bound)#choosing thickness of next column
                
        mutated_balance = np.sum(t - t_ref)
        
        individual_chromosome[bandwidth_left:bandwidth_right+1] = t
        
        return individual_chromosome, mutated_balance, t
        
    def swap_thickness_mutation(self,individual_chromosome, mutation_location, bc):
        
        delta_t_min = bc.ix["Layer thickness"]
        thickness_bound = bc.ix["Thickness Bound"]
        t_bound = np.round(thickness_bound * delta_t_min,1)
        t_0 = bc.ix["minimum thickness"]
        t_max = bc.ix["maximum thickness"]
        
        """
        Randomly choose the swapping location for the mutation operation
        """
        location_possibilities = np.delete(range(0,int(0.5*len(individual_chromosome))),mutation_location)
        swap_location = np.random.choice(location_possibilities)
        
        if swap_location == int(0.5*len(individual_chromosome)):
            swap_location -1
        if swap_location == 0:
            swap_location + 1
        if mutation_location == int(0.5*len(individual_chromosome)):
            mutation_location -1
        if mutation_location == 0:
            mutation_location + 1
        
        """
        Calculate both thickness bounds of the mutation location and swap location
        """
        bound_mutation_left = (individual_chromosome[mutation_location-1]+t_bound,individual_chromosome[mutation_location-1]-t_bound)
        bound_mutation_right = (individual_chromosome[mutation_location+1]+t_bound,individual_chromosome[mutation_location+1]-t_bound)
        bound_mutation_upper = min(bound_mutation_left[0],bound_mutation_right[0],t_max)
        bound_mutation_lower = max(bound_mutation_left[1],bound_mutation_right[1],t_0)
        bound_mutation = bound_mutation_upper - bound_mutation_lower
        
        bound_swap_left = (individual_chromosome[swap_location-1]+t_bound,individual_chromosome[swap_location-1]-t_bound)
        bound_swap_right = (individual_chromosome[swap_location+1]+t_bound,individual_chromosome[swap_location+1]-t_bound)
        bound_swap_upper = min(bound_swap_left[0],bound_swap_right[0],t_max)
        bound_swap_lower = max(bound_swap_left[1],bound_swap_right[1],t_0)
        bound_swap = bound_swap_upper - bound_swap_lower
        """
        Calculate the maximum swap thickness and take it from swap location and add it to mutation location
        """

        swap_location_lower_bound = individual_chromosome[swap_location]-bound_swap_lower
        mutation_location_upper_bound = bound_mutation_upper-individual_chromosome[mutation_location]
        
        max_swap_thickness = min(swap_location_lower_bound, mutation_location_upper_bound)
        individual_chromosome[mutation_location] = individual_chromosome[mutation_location] + max_swap_thickness
        individual_chromosome[swap_location] = individual_chromosome[swap_location] - max_swap_thickness
        
        return individual_chromosome
        