#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:56:05 2018

@author: bart
"""
import pickle
import visuals

#Experiments = ["Ex_109"]

#Experiments = ["Ex_1","Ex_30","Ex_31","Ex_32","Ex_33"] #SLCO1

Experiments = ["Ex_2","Ex_7","Ex_10","Ex_9","Ex_8"] #EXLCO1

#Experiments = ["Ex_11","Ex_13","Ex_15","Ex_12","Ex_14"] #EXLCO2

#Experiments = ["Ex_16","Ex_17","Ex_19","Ex_21","Ex_47"] #DETLCO1

#Experiments = ["Ex_24","Ex_25","Ex_26","Ex_27","Ex_28"] #DETLCO2

# Experiments = ["Ex_53","Ex_54", "Ex_65", "Ex_66", "Ex_67"]

#Experiments = ["Ex_73","Ex_74", "Ex_75", "Ex_76", "Ex_77"]

#Experiments = ["Ex_79","Ex_78","Ex_68","Ex_69","Ex_70", "Ex_71","Ex_72"]

#Experiments = ["Ex_80","Ex_81","Ex_82","Ex_83"]

#Experiments = ["Ex_84","Ex_85","Ex_86", "Ex_87", "Ex_88"] #DETLCM3_Npop

#Experiments = ["Ex_84","Ex_89","Ex_90","Ex_91","Ex_92","Ex_94","Ex_95","Ex_96","Ex_100","Ex_101"] # different mutation heuristics for N = 6 

#Experiments = ["Ex_96","Ex_97", "Ex_98", "Ex_99"]   #DETLCM11.Pm

#Experiments = ["Ex_102"]

VaryingVariable = "Npop"

with open('AlleleDictionaryComplete.pickle','rb') as handle: AlleleDictionaryComplete = pickle.load(handle)

n = 5
t = 3

NeutralAlleleStrengths = visuals.PopulationVisuals.ShowNeutralAlleles(AlleleDictionaryComplete, Experiments, n , t, VaryingVariable)


