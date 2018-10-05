#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:18:34 2018

@author: bart
"""
import numpy as np
import json
import pickle


"""
This file can be used to set up an experiment by linking to the related experiments, constraints and boundary conditions in the database.
"""

"""
#==============================================================================
#                       Choose which Experiment to Run
#==============================================================================
"""
#check the hard-coded EXPERIMENT ID in StorePopulationComposition method!
#check the hard-coded EXPERIMENT ID in ShowTop3CrenellationPatterns method!

ExperimentNumberID = 1  # can be used to select different sets of boundary conditions from the database
np.random.seed(45) #45 used standard



print("Experiment Number "+str(ExperimentNumberID)+" has been started")
import main_script 

"""
Step 1. Collect all boundary conditions and material constants for the experiment chosen
"""

import database_connection
BC = database_connection.Database.RetrieveBoundaryConditions(ExperimentNumberID)
MAT = database_connection.Database.RetrieveMaterial(BC.Material_ID[0])
CONSTRAINTS = database_connection.Database.RetrieveConstraints(BC.Constraint_ID[0])


"""
Step 2. Determine which experimental variables to vary
"""

#if BC.VaryingVariables[0] == 'None':
#    """
#    Run experiment with current paramaters where none of them varies.
#    """
#    pass
#
#else:
#    
#    VaryingVariables = np.array(BC.VaryingVariables[0].split(','))
#    
#    for varyingVariable in VaryingVariables:
#        
#        for variation in range(0,len(BC[str(varyingVariable)])):
#            
#            pass
#            
#            
#                

for Run in range(1,int(BC.NumberOfRuns)+1):     

    print("The algorithm has started run number "+str(Run))

    PopulationComposition, topPatterns, AlleleStrength, PopulationFinal = main_script.MainProgramme.RunGeneticAlgorithm(ExperimentNumberID,BC,MAT,CONSTRAINTS)
    
    print("The algorithm has completed run number "+str(Run))


"""
Build AlleleDictionary from PopulationComposition data, pickle it into folder
"""

AlleleDictionaryComplete = database_connection.Database.StoreAlleleDictionary(ExperimentNumberID, PopulationComposition, BC, CONSTRAINTS)


"""
Store results (PopulationComposition & AlleleDictionaryComplete) in JSON file in folder
"""

database_connection.Database.StoreResultsJSON(PopulationComposition, AlleleDictionaryComplete, ExperimentNumberID)

"""
Bart: still add in method to save results into the master thesis database, both JSON and PopulationComposition readable for Python. Make a method out of this in database_connection.py
"""
print("Experiment Number "+str(ExperimentNumberID)+" has completed")


