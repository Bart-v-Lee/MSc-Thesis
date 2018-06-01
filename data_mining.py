#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 07:25:19 2018

@author: bart
"""


"""
Pseudo-code


Input: problem class variables - n_total, containers per section, range of T, objective function
Input: dataset size, classification algorithm type, 


Method: GenerateDatasets
for range(0,len(dataset))
    Generate genotype pairs of solutions
        evaluate fitness of solutions
        determine better / worse solution and the gap in fitness (%)
        determine the metrics for each solutions
        store metrics in dataset

Method: apply classification algorithm
    split dataset in training and testing data (done in previous method to maintain a single for loop)

    apply data mining techniques to training data to create a model

    test model against the testing data, extract % accuracy


extract rules from the decision tree for different problem classes

"""
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.svm import Linear-SVC


class DataMining:
        
    
    def GenerateDatasets(datasetSize = 600, ClassID, W = 150, delta_a = 1):
        
        import database_connection
        ProblemClassVariables = database_connection.Database.RetrieveProblemClassVariables(ClassID)
        
        nTotalSymmetry = int(ProblemClassVariables.nTotal[0] / 2) # divide by 2 due to symmetry condition
        nTotal = int(ProblemClassVariables.nTotal[0] )
        tDict = ProblemClassVariables.tDictionary[0]
        ObjectiveFunction = ProblemClassVariables.ObjectiveFunction[0]
        nPerSection = ProblemClassVariables.nPerSection[0]
    
        NumberOfSections = int(nTotalSymmetry / nPerSection)
    
        columns = ['ClassID','PairID','Genotype','Fitness','Fitness Gap','Optimal', 'Train data']
        
        for Section in range(1,NumberOfSections+1):
            columns.append("S"+str(Section)+"-tavg")
            columns.append("S"+str(Section)+"-tmax")
            columns.append("S"+str(Section)+"-tmin")
            columns.append("S"+str(Section)+"-tvar")
        
        Dataset = pd.DataFrame(data = None, columns = columns )
        
        
        for PairID in range(0,datasetSize):
            
            # randomly generate 2 genotype solutions 
            """
            Attention: Symmetry constraint still hardcoded
            """            
            
            UniqueThicknessLevels = len(tDict)
            
            ThicknessLevelsSolution1 = np.random.randint(UniqueThicknessLevels,size = nTotalSymmetry)
            ThicknessLevelsSolution2 = np.random.randint(UniqueThicknessLevels,size = nTotalSymmetry)

            
            GenotypeSolution1 = []
            GenotypeSolution2 = []

            
            for Thickness in ThicknessLevelsSolution1:
                GenotypeSolution1 = np.append(GenotypeSolution1,float(tDict[str(Thickness)]))
            for Thickness in ThicknessLevelsSolution2:
                GenotypeSolution2 = np.append(GenotypeSolution2,float(tDict[str(Thickness)]))

            # Apply symmetry condition 
            
            GenotypeSolution1 = np.append(GenotypeSolution1, np.flipud(GenotypeSolution1))
            GenotypeSolution2 = np.append(GenotypeSolution2, np.flipud(GenotypeSolution2))

            # transform 2 solutions into chromosomes using ConstructChromosomeGenotype
            
            import crenellation
            ChromosomeSolution1 = crenellation.CrenellationPattern.ConstructChromosomeGenotype(GenotypeSolution1, nTotal, W, delta_a)
            ChromosomeSolution2 = crenellation.CrenellationPattern.ConstructChromosomeGenotype(GenotypeSolution2, nTotal, W, delta_a)
            
            
            # evaluate the fitness (using fitness function method from GA) & determine better / worse solution
            
            import genetic_algorithm
            
            BC = database_connection.Database.RetrieveBoundaryConditions(ExperimentNumberID)
            MAT = database_connection.Database.RetrieveMaterial(BC.Material_ID[0])
            
            FitnessSolution1, FatigueCalculations = genetic_algorithm.GeneticAlgorithm.EvaluateFitnessFunction(ObjectiveFunction, ChromosomeSolution1, BC.S_max[0], BC.a_0[0], BC.a_max[0], BC.Delta_a[0],MAT.C[0],MAT.m[0] )
            FitnessSolution2, FatigueCalculations = genetic_algorithm.GeneticAlgorithm.EvaluateFitnessFunction(ObjectiveFunction, ChromosomeSolution2, BC.S_max[0], BC.a_0[0], BC.a_max[0], BC.Delta_a[0],MAT.C[0],MAT.m[0] )
            
            if FitnessSolution1 > FitnessSolution2:
                Solution1Optimal = True
                Solution2Optimal = False
                FitnessGap = ((FitnessSolution1 - FitnessSolution2) / FitnessSolution1) * 100
            else:
                Solution1Optimal = False
                Solution2Optimal = True
                FitnessGap = ((FitnessSolution2 - FitnessSolution1) / FitnessSolution2) * 100



            # calculate the (geo)metrics for each solution in a dictionary
            
            data1 = {'ClassID': ClassID, "PairID": PairID, "Genotype": [GenotypeSolution1], "Fitness": FitnessSolution1, "Fitness Gap": FitnessGap, "Optimal": Solution1Optimal}
            data2 = {'ClassID': ClassID, "PairID": PairID, "Genotype": [GenotypeSolution2], "Fitness": FitnessSolution2, "Fitness Gap": FitnessGap, "Optimal": Solution2Optimal}
            
            MetricDataSolution1 = {}
            MetricDataSolution2 = {}
            
            for Section in range(1,NumberOfSections+1):
                tavg1 = np.average(GenotypeSolution1[nPerSection*(Section-1):nPerSection*Section])
                tmax1 = np.max(GenotypeSolution1[nPerSection*(Section-1):nPerSection*Section])
                tmin1 = np.min(GenotypeSolution1[nPerSection*(Section-1):nPerSection*Section])
                tvar1 = np.var(GenotypeSolution1[nPerSection*(Section-1):nPerSection*Section])
                
                tavg2 = np.average(GenotypeSolution2[nPerSection*(Section-1):nPerSection*Section])
                tmax2 = np.max(GenotypeSolution2[nPerSection*(Section-1):nPerSection*Section])
                tmin2 = np.min(GenotypeSolution2[nPerSection*(Section-1):nPerSection*Section])
                tvar2 = np.var(GenotypeSolution2[nPerSection*(Section-1):nPerSection*Section])
                
                data1['S'+str(Section)+'-tavg'] = tavg1
                data1['S'+str(Section)+'-tmax'] = tmax1
                data1['S'+str(Section)+'-tmin'] = tmin1
                data1['S'+str(Section)+'-tvar'] = tvar1
                
                data2['S'+str(Section)+'-tavg'] = tavg2
                data2['S'+str(Section)+'-tmax'] = tmax2
                data2['S'+str(Section)+'-tmin'] = tmin2
                data2['S'+str(Section)+'-tvar'] = tvar2

            # randomly decide whether a pair will be used for training or testing the classification model (later on)
            
            
            RandomNumber = np.random.uniform(0.0,1.0)
            if RandomNumber <= 0.7:
                TrainData = 1
                data1["Train data"] = int(TrainData)
                data2["Train data"] = int(TrainData)

            else:
                TrainData = 0
                data1["Train data"] = int(TrainData)
                data2["Train data"] = int(TrainData)
            
            # store all data in dataset dataframe
        

            DatasetAdd1 = pd.DataFrame(data = data1)
            DatasetAdd2 = pd.DataFrame(data = data2)

            Dataset = Dataset.append(DatasetAdd1, ignore_index = True)
            Dataset = Dataset.append(DatasetAdd2, ignore_index = True)
            print("Dataset expaned...pair number "+str(PairID))
            
            return Dataset
    
    def ApplyClassificationMethod(Dataset, ClassificationMethod = "Set 1"):
    

            
        if ClassificationMethod = "Set 1":
            """
            Linear SVM classification method
            """
            
    
    
    