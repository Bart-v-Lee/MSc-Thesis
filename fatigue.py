#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:17:40 2017

@author: Bart van der Lee
@project: MSc thesis 
"""

import numpy as np
import pandas as pd


class FatigueCalculations:
    
    def __init__(self):
        pass
    
    
    def CalculateChromosomeArea():
        """
        Calculates the total area of a given chromosome
        """
        
        
        pass
    
    
    def CalculateFatigueLife(Chromosome, S_max, a_0, a_max, delta_a, C, m): #previously fatigue_calculations
        """
        Method used to evaluate the fatigue crack growth life of a single solution, or in GA terms "individual".
        """
        
        # Calculate several variables needed for the fatigue life calculations
        
        delta_a_meters = delta_a / 1000
        total_a = a_max - a_0

        
        # Import empty dataframe to store the Fatigue Calcutions for the current solution
        
        import database_connection  
        FatigueCalculations = database_connection.Database.RetrieveFatigueDataframe()
        
        # Create the necessary number of rows in the Dataframe for each crack growth increment 'da'
        
        NumberOfCrackIncrements = int(total_a / delta_a)
        FatigueCalculations['a'] = np.linspace(a_0,a_max,num = NumberOfCrackIncrements).round(decimals=2)
        a_meters = FatigueCalculations['a'] / 1000
        
        """
        Convert thickness pattern into dataframe with containers
        """
        
#        thickness_pattern_dataframe = cren_design.create_dataframe(fatigue_lifetime, thickness_pattern, bc,m2)
#        thickness_pattern = thickness_pattern_dataframe[1]
#        print(thickness_pattern_dataframe)
# 
        """
        Start evaluating crack growth equations
        """
        
        """
        Evaluate the area in front of the crack
        """
        
        import crenellation
        
        # use both entire dataframes because the computation can be done much quicker
        
        FatigueCalculations = crenellation.CrenellationPattern.CalculateAreaInFrontOfCrack(FatigueCalculations, Chromosome)
        
        
        """
        Old calculation of area
        """
#            print("starting with area calculation")
        fatigue_lifetime = crenellation.CalculateAreaInFrontOfCrack(self, thickness_pattern, fatigue_lifetime, bc)
#            print("area inserted into dataframe")



        """
        Evaluate the effective stress Sigma_Eff
        """
        FatigueCalculations.sigma_eff = S_max / (2 * FatigueCalculations.area)  
        
        """
        Evaluate the Stress Intensity Factor K
        """
        FatigueCalculations.K = FatigueCalculations.sigma_eff * np.sqrt(np.pi*a_meters) 
        
        """
        Calculate the crack growth rate used the Paris relation
        """
        
        FatigueCalculations.dadN = C*(FatigueCalculations.K)**m
        
        """
        Calculate the fatigue life increment dN for the given crack growth increment da
        """
        
        FatigueCalculations.dN = delta_a_meters / FatigueCalculations.dadN
        
#            print("iterating over dN per crack step")
        """
        Calculate the cumulative fatigue life by summating the previous fatigue life increments dN for every step
        """
            
        for j in range(0,len(fatigue_lifetime)):
            fatigue_lifetime['N'][j+1] = fatigue_lifetime['N'][j] + fatigue_lifetime['dN'][j]
#                cren_design["sigma_iso"][j] = S_max / (2 * (half_width - thickness_pattern["Width"][cren_design.index[j]]) * np.sum(thickness_pattern["thickness"]))
        
        """
        Calculate a comparison measure Beta (Sigma_Eff / Sigma_Applied) - to discuss with Calvin. Not critical to the calculations.
        """
#            cren_design["sigma_iso"] = cren_design["sigma_eff"] / np.sqrt(1-(   /   1 ))
        fatigue_lifetime["beta"] = fatigue_lifetime["sigma_eff"] / sigma_applied
        
        """
        Calculate the total fatigue life by summating the fatigue crack growth increments dN for every step of da
        """
        N = sum(fatigue_lifetime['dN'])
        population["Fitness"][i] = N
        
        """
        Calculate control measures to check if constraints are upheld.
        Upper bound resembles the upper thickness level t_max
        Lower bound resembles the lower thickness level t_min
        Balance resembles the equal weight criterion according to t_ref and W
        """
#            population["Cren Design"][i] = cren_design #turned off for brute force method
        population["Balance"][i] = np.sum(thickness_pattern["thickness"] - t_ref)
        population["Upper Bound"][i] = np.count_nonzero(thickness_pattern["thickness"]>t_max)
        population["Lower Bound"][i] = np.count_nonzero(thickness_pattern["thickness"]<t_min)
        
        print("Individual ",i," has been evaluated")
            

        return population

    """
    #==============================================================================
    # END - further code not necessarily needed -  Old code not yet to be thrown away
    #==============================================================================
    """
    def CalculateFatigueLife_OLD_version1(Chromosome, S_max, a_0, a_max, Delta_a, C, m): #previously fatigue_calculations
        """
        Method used to evaluate the fatigue crack growth life's of a population of solutions
        """
        
        """
        Assign fatigue calculation variables from boundary conditions
        """
        pop_size = len(population)
        S_max = bc.ix["Smax"]         
        C = m2.ix["C"]
        m = m2.ix["m"]
        t_ref = bc.ix["Reference thickness"]
        t_max = bc.ix["maximum thickness"]
        t_min = bc.ix["minimum thickness"]
        delta_a = bc.ix["crack step size"]
        delta_a_meters = delta_a / 1000
        delta_a = bc.ix["crack step size"]
        a_max = bc.ix["Max crack length"]
        a_0 = bc.ix["Initial crack length"]
        total_a = a_max - a_0
        half_width = int(bc.ix["Width"] / 2 )
        
        """
        Create dataframe for fatigue calculations 
        """
        
        size = int(total_a / delta_a)
        index = range(0,size)
        list = {'width', 'a','N','dadN','K','sigma_eff','area','dN', 'sigma_iso', 'beta'}
        array = np.zeros((size,len(list)), dtype = 'float')
        fatigue_lifetime = pd.DataFrame(data=array,index=index,columns=list, dtype='float')
        fatigue_lifetime['a'] = np.linspace(a_0,a_max,num = size).round(decimals=2)
        sigma_applied = 1
        
        for i in range(1,pop_size+1): 
            cren_design = crenellation(bc,m2)
            
            """
            Find thickness pattern for current individual i inside chromosome
            """
            
            thickness_pattern = population["Chromosome"][i]
            
            """
            Convert thickness pattern into dataframe with containers
            """
            
            thickness_pattern_dataframe = cren_design.create_dataframe(fatigue_lifetime, thickness_pattern, bc,m2)
            thickness_pattern = thickness_pattern_dataframe[1]
            print(thickness_pattern_dataframe)
            a = fatigue_lifetime['a']
            a_meters = a / 1000
            
            """
            Start evaluating crack growth equations
            """
            
            """
            Evaluate the area in front of the crack
            """
#            print("starting with area calculation")
            fatigue_lifetime = crenellation.cal_cren_area(self, thickness_pattern, fatigue_lifetime, bc)
#            print("area inserted into dataframe")
            """
            Evaluate the effective stress Sigma_Eff
            """
            fatigue_lifetime.sigma_eff = S_max / (2 * fatigue_lifetime.area)  
            """
            Evaluate the Stress Intensity Factor K
            """
            fatigue_lifetime.K = fatigue_lifetime.sigma_eff * np.sqrt(np.pi*a_meters) 
            """
            Calculate the crack growth rate used the Paris relation
            """
            fatigue_lifetime['dadN'] = C*(fatigue_lifetime['K'])**m
            """
            Calculate the fatigue life increment dN for the given crack growth increment da
            """
            fatigue_lifetime['dN'] = delta_a_meters / fatigue_lifetime['dadN']
            
#            print("iterating over dN per crack step")
            """
            Calculate the cumulative fatigue life by summating the previous fatigue life increments dN for every step
            """
                
            for j in range(0,len(fatigue_lifetime)):
                fatigue_lifetime['N'][j+1] = fatigue_lifetime['N'][j] + fatigue_lifetime['dN'][j]
#                cren_design["sigma_iso"][j] = S_max / (2 * (half_width - thickness_pattern["Width"][cren_design.index[j]]) * np.sum(thickness_pattern["thickness"]))
            
            """
            Calculate a comparison measure Beta (Sigma_Eff / Sigma_Applied) - to discuss with Calvin. Not critical to the calculations.
            """
#            cren_design["sigma_iso"] = cren_design["sigma_eff"] / np.sqrt(1-(   /   1 ))
            fatigue_lifetime["beta"] = fatigue_lifetime["sigma_eff"] / sigma_applied
            
            """
            Calculate the total fatigue life by summating the fatigue crack growth increments dN for every step of da
            """
            N = sum(fatigue_lifetime['dN'])
            population["Fitness"][i] = N
            
            """
            Calculate control measures to check if constraints are upheld.
            Upper bound resembles the upper thickness level t_max
            Lower bound resembles the lower thickness level t_min
            Balance resembles the equal weight criterion according to t_ref and W
            """
#            population["Cren Design"][i] = cren_design #turned off for brute force method
            population["Balance"][i] = np.sum(thickness_pattern["thickness"] - t_ref)
            population["Upper Bound"][i] = np.count_nonzero(thickness_pattern["thickness"]>t_max)
            population["Lower Bound"][i] = np.count_nonzero(thickness_pattern["thickness"]<t_min)
            
            print("Individual ",i," has been evaluated")
            

        return population

        
#    
#    def single_pattern_fatigue(self, chromosome, bc, m2):
#        
#        """
#        Method used to evaluate the fatigue crack growth life of a single crenellation pattern
#        """
#        pop_size = int(bc.ix["Population size"])
#        S_max = bc.ix["Smax"]   
#        S_min = bc.ix["Smin"]                   #Newton             
#        C = m2.ix["C"]
#        m = m2.ix["m"]
#        t_ref = bc.ix["Reference thickness"]
#        t_max = bc.ix["maximum thickness"]
#        t_min = bc.ix["minimum thickness"]
#        delta_a = bc.ix["crack step size"]
#        delta_a_meters = delta_a / 1000
#        delta_a = bc.ix["crack step size"]
#        a_max = bc.ix["Max crack length"]
#        a_0 = bc.ix["Initial crack length"]
#        total_a = a_max - a_0
#        half_width = int(bc.ix["Width"] / 2)
#        """
#        Create dataframe for fatigue calculations
#        """
#        size = int(total_a / delta_a)
#        index = range(0,size)
#        list = {'width', 'a','N','dadN','delta_K','sigma_eff_min','sigma_eff_max', 'delta_sigma_eff','area','dN', 'sigma_iso', 'beta'}
#        array = np.zeros((size,len(list)), dtype = 'float')
#        lifetime = pd.DataFrame(data=array,index=index,columns=list, dtype='float')
#        lifetime['a'] = np.linspace(a_0,a_max,num = size).round(decimals=2)
#        sigma_applied = 1
#        
#        cren_design = crenellation(bc,m2)
#        thickness_pattern = chromosome
#        cren_pattern = cren_design.create_dataframe(lifetime, thickness_pattern, bc,m2)
#        cren_design = cren_pattern[0]
#        thickness_pattern = cren_pattern[1]
#        a = cren_design['a']
#        a_meters = a / 1000
#        
#        """
#        Start evaluating crack growth equations
#        """
#        
##            print("starting with area calculation")
#        cren_design = crenellation.cal_cren_area(self,  thickness_pattern, cren_design, bc)
##            print("area inserted into dataframe")
#        
#        cren_design.sigma_eff_max = S_max /  (cren_design.area) #removed the division by two
#        cren_design.sigma_eff_min = S_min / (cren_design.area) #removed the division by two
#
#        cren_design.delta_sigma_eff = cren_design.sigma_eff_max - cren_design.sigma_eff_min
#        cren_design.delta_K = cren_design.delta_sigma_eff * np.sqrt(np.pi*a_meters) 
#        
##        cren_design.K_max = cren_design.sigma_eff_max * np.sqrt(np.pi*a_meters) 
##        cren_design.K_min = cren_design.sigma_eff_min * np.sqrt(np.pi*a_meters) 
#
##        cren_design.delta_K_2 = cren_design.K_max - cren_design.K_min
#
#        cren_design['dadN'] = C*(pow(cren_design['delta_K'],m))
#        cren_design['dN'] = delta_a_meters / cren_design['dadN']
#        
##            print("iterating over dN per crack step")
#
#            
#        for j in range(0,len(cren_design)):
#            cren_design['N'][j+1] = cren_design['N'][j] + cren_design['dN'][j]
##                cren_design["sigma_iso"][j] = S_max / (2 * (half_width - thickness_pattern["Width"][cren_design.index[j]]) * np.sum(thickness_pattern["thickness"]))
#        
#
##            cren_design["sigma_iso"] = cren_design["sigma_eff"] / np.sqrt(1-(   /   1 ))
#        cren_design["beta"] = cren_design["delta_sigma_eff"] / sigma_applied
#
#        
#        return cren_design
#        


        
#    def rand_thickness_at_a(self,N,bc,m2,t,a, a_ref, a_prev, a_ref_prev):    
#
#        
#        a_current = a_prev[1][0]  #mm  #crack length from previous cycle N, needs to have SAME NUMBER OF DECIMALS as Delta_X
#        a_current_ref = a_ref_prev[1][0]
#        
#        t_current = t[1][0] #thickness at point a_current
#        t_current_ref = t_ref
#        
#
#
#        self.a_current = a_current
#        self.a_current_ref = a_current_ref
#        return [a_current, a_current_ref, t_current, t_current_ref, a_current_meters, a_current_meters_ref]
#        

#    def SIF(self, t, a, a_prev, a_ref, a_ref_prev, bc, m2):
#
#        W = int(bc.ix["Width"])                    #mm
#        half_width = np.int(W/2)                     #mm
#        t_ref = bc.ix["Reference thickness"]
##        R = S_min / S_max                            #Stress ratio
##        S_mean = (S_max + S_min)/2                   #mean stress level
##        S_min = bc.ix["Smin"]                      #Newton
#        Y = 1                                        #shape factor , default = 1
#        a_current = a_prev[1][0]
#        a_current_ref = a_ref_prev[1][0]
#        t_current = t[1][0]
#        t_current_ref = t_ref
#        a_current_meters = a_current / 1000
#        a_current_meters_ref = a_current_ref / 1000
#        """
#        area of uncracked plate in front of the crack; calculation through integral over area beneath thickness
#        """
#        area_t = (t_current * np.sqrt(1-(np.divide(a_current,half_width)**2))* half_width) - (t_current * np.sqrt(1-(np.divide(a_current,a_current)**2))*a_current)    #evaluates the integral of a until 0.5 W for a unit thickness plate of the same thickness as found around the crack tip after cycle N-1
#        area_t_ref = (t_current_ref * np.sqrt(1-(np.divide(a_current_ref,half_width)**2))*half_width) - (t_current_ref * np.sqrt(1-(np.divide(a_current_ref,a_current_ref)**2))*a_current_ref)
#        """
#        calculation of the effective stress at the crack tip
#        """
#        sigma_eff = S_max / area_t
#        sigma_eff_ref = S_max / area_t_ref
#        """
#        calculation of the stress intensity factor around the crack tip at a given crack length
#        """
#        K =  Y * sigma_eff * np.sqrt(np.pi*a_current_meters) #MPa sqrt(m)
#        K_ref =  Y * sigma_eff_ref * np.sqrt(np.pi*a_current_meters_ref) 
#
##        if K <= Kth:
##            print( "stress intensity below fatigue limit:", K, "with a barrier Kth of", Kth)
#
#        self.K = K
#        self.K_ref = K_ref
#        
#    def crack_growth(self, K, K_ref, N, bc, m2, a, a_ref, a_prev, a_ref_prev):
#
#        delta_a =  (C*K**m)*1000 #mm / cycle
#        delta_a_ref =  (C*K_ref**m)*1000 #mm / cycle
#        
#        a[1][0] = a_prev[1][0] + delta_a #new crack length after 1 cycle N
#        a[1][1] = delta_a
#        a[1][2] = K
#
#        a_ref[1][0] = a_ref_prev[1][0] + delta_a_ref #new crack length after 1 cycle N
#        a_ref[1][1] = delta_a_ref
#        a_ref[1][2] = K_ref
#
#        return [a, a_ref]
#        
