#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:39:57 2018

@author: Bart van der Lee
@project: MSc thesis 
"""

class Database:
    
    def __init__():
        """
        """
        
        
        
        
        pass
        
    def RetrieveMaterial(MaterialID):
        """
        Retrieves the Material constants for a chosen Material ID
        """
        
        pass
        
        
    def RetrievePopulationDataframe():
        """
        Method imports an empty template dataframe from the database for a population
        """
        
        conn = sqlite3.connect("database_thesis.db")
        cur =  conn.cursor()
        PopulationDataframe = pd.read_sql_query("Select * from Population;", conn) 
        conn.close

        return PopulationDataframe
    
    def RetrieveBoundaryConditions(ExperimentNumberID):
        """
        Retrieves the Boundary Conditions record from the database for a given Experiment Number
        """
        
        delta_x = 
        W = 
        N_pop = 
        t_dict = 
        
        delta_x, W, N_pop, t_dict, SeedSettings, SeedNumber, NumberOfRuns, NumberOfGenerations, PopStatisticsDict, Rs, Pc, Pm, S_max, a_0, a_max, delta_a,C,m
        
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
        
        
        pass
    
    def RetrieveSeedShape(self, SeedNumber, delta_x, W):
        """
        Retrieves a seed shape from
        """
        
        
        
        
        
        pass