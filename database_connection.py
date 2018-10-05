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
        
    def RetrieveProblemClassVariables(ClassID):
        
        
        conn = sqlite3.connect("database_thesis.db")
        cur = conn.cursor()
        
        ProblemClassVariables = pd.read_sql_query("SELECT * FROM ProblemClasses WHERE ClassID = (?)", conn, params = (str(ClassID),) )

        
        # Transform T_dict string into a dictionary and place back into the BoundaryConditions dataframe

        tDictionary = json.loads(ProblemClassVariables.tDictionary[0])
        ProblemClassVariables.at[0,'tDictionary'] = tDictionary
        conn.close
    
        return ProblemClassVariables
    
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
<<<<<<< HEAD
    
    def StorePopulationCompositionDictionary(PopulationComposition, ExperimentNumberID):
        
        
        with open('PopulationCompositionComplete.pickle','rb') as handle: PopulationCompositionComplete = pickle.load(handle)

        PopulationCompositionComplete["Ex_"+str(ExperimentNumberID)] = PopulationComposition
        
        with open('PopulationCompositionComplete.pickle','wb') as handle: pickle.dump(PopulationCompositionComplete, handle, protocol = pickle.HIGHEST_PROTOCOL)

        
    
    def StoreAlleleDictionary(ExperimentNumberID, PopulationComposition, BC, CONSTRAINTS):
        """
        This method creates a dictionary AlleleDictionary with AlleleStrength values from PopulationComposition for a given experiment and stores this inside a dictionary 
        AlleleDictionaryComplete in which data is stored from different experiments.
        This dictionary, in contrary to PopulationComposition, is suitable for analysis accross experiments
        
        Could be improved further by storing the pickle file in the SQLite database, instead as a file in the folder. Yet, at this point not necessary.
        """
        
        #create dictionary from PopulationComposition
        
        ExperimentDict = {}
        AlleleDict = {}
        GenerationDict = {}
        GeneDict = {}
        maxTindex = int(max(BC.T_dict[0]))
        
        if CONSTRAINTS.Plate_Symmetry[0] == 'True':
            n_total = int(BC.n_total[0] /2)
        else:
            n_total = int(BC.n_total[0])
            
        t_total = len(BC.T_dict[0])
            
        
        for gene in range(0,n_total):
            
            for allele in range(0,len(BC.T_dict[0])):
                
                for generation in range(0,int(BC.NumberOfGenerations[0])):
                    
                    GenerationDict["Gen_"+str(generation)] = {"Fitness": PopulationComposition["Gen "+str(generation)][2][maxTindex-allele][gene], "Gen": generation , "Evaluations":  PopulationComposition["Gen "+str(generation)][3]}
                    
                ExperimentDict["Ex_"+str(ExperimentNumberID)] = GenerationDict # 
                GenerationDict = {}
        
                AlleleDict["Allele_"+str(allele)] = ExperimentDict
                ExperimentDict = {}
        
            GeneDict["Gene_"+str(gene)] = AlleleDict
            AlleleDict = {}
        
                
                    
        AlleleDictionary = {"n_"+str(n_total): {"t_"+str(t_total): GeneDict}}
        #print(AlleleDictionary)
        
        #read the pickled AlleleDictionary file
        
        try:
            with open('AlleleDictionaryComplete.pickle','rb') as handle: AlleleDictionaryComplete = pickle.load(handle)
            
            #store AlleleDictionary from current experiment in the existing dictionary
            
            print("reading pickle has succeeded")
            
                
            if "n_"+str(n_total) in AlleleDictionaryComplete.keys():
               
                if "t_"+str(t_total) in AlleleDictionaryComplete["n_"+str(n_total)].keys():
                    
                    for gene in range(0,n_total):
                        
                        for allele in range(0,len(BC.T_dict[0])):
                            
                            if "Ex_"+str(ExperimentNumberID) in AlleleDictionaryComplete["n_"+str(n_total)]["t_"+str(t_total)]["Gene_"+str(gene)]["Allele_"+str(allele)].keys():
                                
                                AlleleDictionaryComplete["n_"+str(n_total)]["t_"+str(t_total)]["Gene_"+str(gene)]["Allele_"+str(allele)]["Ex_"+str(ExperimentNumberID)] = AlleleDictionary["n_"+str(n_total)]["t_"+str(t_total)]["Gene_"+str(gene)]["Allele_"+str(allele)]["Ex_"+str(ExperimentNumberID)]


                            else:
                                AlleleDictionaryComplete["n_"+str(n_total)]["t_"+str(t_total)]["Gene_"+str(gene)]["Allele_"+str(allele)]["Ex_"+str(ExperimentNumberID)] = AlleleDictionary["n_"+str(n_total)]["t_"+str(t_total)]["Gene_"+str(gene)]["Allele_"+str(allele)]["Ex_"+str(ExperimentNumberID)]
                                        
                
                else:
                    # if t not yet present
                    AlleleDictionaryComplete["n_"+str(n_total)]["t_"+str(t_total)] = AlleleDictionary["n_"+str(n_total)]["t_"+str(t_total)]
        
            else:
                AlleleDictionaryComplete["n_"+str(n_total)] = AlleleDictionary["n_"+str(n_total)]
                    
                    
            #pickle the dictionary and save in folder

            
            with open('AlleleDictionaryComplete.pickle','wb') as handle: pickle.dump(AlleleDictionaryComplete, handle, protocol = pickle.HIGHEST_PROTOCOL)

            
        except:
            AlleleDictionaryComplete = AlleleDictionary
            print("reading pickle has failed, results have not been pickled")
        

        return AlleleDictionaryComplete
        
    def StoreResultsJSON(PopulationComposition, AlleleDictionaryComplete, ExperimentNumberID):
        """
        This method stores the optimization results in JSON format such that it can be read by Javascript for the visualisations in D3.
        """
        import database_connection
        BC = database_connection.Database.RetrieveBoundaryConditions(ExperimentNumberID)
        n_total = BC.n_total[0]
        t_total = len(BC.T_dict[0])
            
        with open('result_Ex_'+str(ExperimentNumberID)+'_n_'+str(n_total)+'_t_'+str(t_total)+'.json', 'w') as fp:
            json.dump(PopulationComposition, fp, cls=JSONEncoder)
            
        with open('AlleleDictionaryComplete.json','w') as fp:
            json.dump(AlleleDictionaryComplete, fp, cls=JSONEncoder)



class JSONEncoder(json.JSONEncoder):
    
    def default(self, obj):
        """
        This method determines which json encode method should be used, depending on the data structure of the object (dataframe, dictionary, list, etc)
        """
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
=======
    
>>>>>>> parent of b76d0c9... Update 3.0
