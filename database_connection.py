#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:39:57 2018

@author: Bart van der Lee
@project: MSc thesis 
"""

import pandas as pd
import sqlite3
import json
import numpy as np

class Database:
    
    def __init__():
        """
        """

        pass
        
    def RetrieveMaterial(Material_ID):
        """
        Retrieves the Material constants for a chosen Material ID
        """
        conn = sqlite3.connect("database_thesis.db")
        cur =  conn.cursor()
        MaterialConstants = pd.read_sql_query("SELECT * FROM Materials WHERE id = (?);", conn, params = (str(Material_ID),) ) 
        
        conn.close
        
        return MaterialConstants
        
        
    def RetrieveConstraints(Constraint_ID):
        """
        Retrieves the Material constants for a chosen Material ID
        """
        conn = sqlite3.connect("database_thesis.db")
        cur =  conn.cursor()
        ExperimentConstraints = pd.read_sql_query("SELECT * FROM Constraints WHERE Constraint_ID = (?);", conn, params = (str(Constraint_ID),) ) 
        
        conn.close
        
        return ExperimentConstraints
        
    def RetrieveChromosomeDataframe():
        """
        Retrieves the Material constants for a chosen Material ID
        """
        conn = sqlite3.connect("database_thesis.db")
        cur =  conn.cursor()
        Chromosome = pd.read_sql_query("SELECT * FROM Crenellation;", conn ) 
        
        conn.close
        
        return Chromosome
        
        
    def RetrievePopulationDataframe(N_pop):
        """
        Method imports an empty template dataframe from the database for a population. 
        Maximum population size N_pop is constrained by the number of empty records in the database table Population.
        Increase the number of records in Population table within the database_thesis.db if a larger population size is necessary.
        """
        
        conn = sqlite3.connect("database_thesis.db")
        cur =  conn.cursor()
        PopulationDataframe = pd.read_sql_query("Select * from Population;", conn) 
        
        # truncate the Population Dataframe to the provided population size N_pop
        
        PopulationDataframe = PopulationDataframe[1:N_pop+1]
        
        conn.close

        return PopulationDataframe
    
    def RetrieveBoundaryConditions(ExperimentNumberID):
        """
        Retrieves the Boundary Conditions record from the database for a given Experiment Number
        """
        
        conn = sqlite3.connect("database_thesis.db")
        cur =  conn.cursor()
        BoundaryConditions = pd.read_sql_query('SELECT * FROM Boundary_Conditions WHERE Experiment_ID = (?);', conn, params = (ExperimentNumberID,) ) 
        
        
        # Transform T_dict string into a dictionary and place back into the BoundaryConditions dataframe

        T_dict = json.loads(BoundaryConditions.T_dict[0])
        BoundaryConditions.at[0,'T_dict'] = T_dict
        
        conn.close
        
        return BoundaryConditions
    
    def RetrieveSeedShape(SeedNumber):
        """
        Retrieves a seed shape from the database_thesis.db
        """
        conn = sqlite3.connect('database_thesis.db')
        cur = conn.cursor()
        
        GenotypeSeedDf = pd.read_sql_query('SELECT * FROM Seed_Designs WHERE id = (?);', conn, params = (SeedNumber,) )
        
        # Transform Genotype string into an array 
        
        GenotypeSeed = np.array(GenotypeSeedDf.Genotype[0].split(','))
        
        conn.close
        
        return GenotypeSeed
    
    
    def RetrieveFatigueDataframe():
        """
        Retrieves the Material constants for a chosen Material ID
        """
        conn = sqlite3.connect("database_thesis.db")
        cur =  conn.cursor()
        FatigueDataframe = pd.read_sql_query("SELECT * FROM Fatigue_calculations;", conn) 
        
        conn.close
        
        return FatigueDataframe
    