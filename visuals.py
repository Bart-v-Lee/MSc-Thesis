#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:18:33 2017

@author: Bart van der Lee
"""

import matplotlib.pyplot as pp
from matplotlib import rc
from matplotlib.colors import ListedColormap

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import Heatmap

import numpy as np
import pandas as pd

#pp.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
#params = {'text.usetex': True,
#          'font.size': 11,
#          'font.family': 'lmodern',
#          'text.latex.unicode': True,
#          }
#pp.rcParams.update(params)
        
import matplotlib.animation as animation
import matplotlib.ticker as mtick


ref_data = pd.ExcelFile('/Users/Bart/Google Drive/6. Education/Master/4. MSc Thesis/Python code/classes/version2/reference_study_experimental_data.xlsx')
Uz_crenellation_right = ref_data.parse('Uz (2008) crennelated right')
Uz_crenellation_left = ref_data.parse('Uz (2008) crennelated left')
Huber_cren_prediction =  ref_data.parse('Huber (2009) prediction')
#NASA_flat_sheet =  ref_data.parse('NASA (2001) flat sheet (Ti-6..)')
#Uz_uniform_left = ref_data.parse('Uz (2008) uniform left')
Uz_uniform_right = ref_data.parse('Uz (2008) uniform right')


class FatigueVisuals:
    
    def ShowCrenellationPattern(Chromosome):
        """
        This method shows the Crenellation pattern when a chromosome is provided
        """
        
        
        pass
    
    def ShowFatigueOverview():
        """
        This method shows the Fatigue Calculations overview when a solution is provided
        Instead of storing the FatigueCalculations from the GA calculations, this method re-evaluates the fatigue life by performing the calculation for the given Chromosome
        """
        
        
        
        
        
        
        pass
        
    def ShowTop3CrenellationPatterns(PopulationFinal, PopulationInitial):
        """
        This method plots the top 3 crenellation patterns in the initial and the final population
        """
        PopulationFinal = PopulationFinal.reset_index()
        
        PopulationInitialRanked = PopulationInitial.sort_values("Fitness", ascending=False, kind = 'mergesort')
        PopulationInitialRanked = PopulationInitialRanked.reset_index()
        
        ExperimentNumberID = 1

        import database_connection
        BC = database_connection.Database.RetrieveBoundaryConditions(ExperimentNumberID)
        MaxThickness = float(BC.T_dict[0][str(len(BC.T_dict)+1)])
        
        fig, axes = pp.subplots(nrows=2, ncols=3, figsize=(15,6))
        
        for i in range(0,3):
            
            # Final Population Plots 
            
            IndividualNumber = i + 1
                    
            axes[1,i].plot(PopulationFinal.Chromosome[IndividualNumber].Thickness)
            axes[1,i].fill_between(PopulationFinal.Chromosome[IndividualNumber].Width, 0, PopulationFinal.Chromosome[IndividualNumber].Thickness, facecolor='green')
            axes[1,i].set_ylim([0,MaxThickness])
            
            # Initial Population Figures
            
            axes[0,i].plot(PopulationInitialRanked.Chromosome[IndividualNumber].Thickness)
            axes[0,i].fill_between(PopulationInitialRanked.Chromosome[IndividualNumber].Width, 0, PopulationInitialRanked.Chromosome[IndividualNumber].Thickness)
            axes[0,i].set_ylim([0,MaxThickness])

                
        axes[1,1].set_title('Top 3 Crenellation Patterns within the Final Population')
        
        axes[0,1].set_title('Top 3 Crenellation Patterns within the Initial Population')
        
        
        pp.tight_layout()
        pp.show()
        
        
    
class PopulationVisuals:
    

    def ShowAlleleStrengthComposition(PopulationComposition, NumberOfGenerations):
        """
        This method shows the strength of an allele in time (e.g. generations)
        """
        
        # Import the boundary conditions and constraints

        import database_connection
        ExperimentNumberID = 1
        BC = database_connection.Database.RetrieveBoundaryConditions(ExperimentNumberID)
        CONSTRAINTS = database_connection.Database.RetrieveConstraints(BC.Constraint_ID[0])
        
        #figure = pp.figure(figsize = (15,15))

        # Determine the number of alleles and genes given the constraints and boundary conditions
        
        NumberOfAlleles = len(BC.T_dict[0])
        if CONSTRAINTS.Plate_Symmetry[0] == "True":
            NumberOfGenes = int(BC.n_total[0] / 2   )
        else:
            NumberOfGenes = BC.n_total[0]

        #NumberOfBuildingBlocks = NumberOfGenes * NumberOfAlleles
        #BuildingBlocks = ()
        pp.figure(1, figsize = (15,15))
        figure2, axes = pp.subplots(nrows=NumberOfAlleles-1, ncols=NumberOfGenes, figsize=(8,8))
        
        colors = ['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#000000','#800000']

        # Iterate for each Allele, Gene combination and plot the Allele Strength for each generation 

        BuildingBlockNumber = -1

        for Gene in range(0,NumberOfGenes):
        
            for Allele in range(0,NumberOfAlleles):
                if Allele == 0:
                    continue
                
                AlleleArray = np.zeros((NumberOfAlleles, NumberOfGenes))
                AlleleArray[NumberOfAlleles-1-Allele:3,Gene] = 1.0
                BuildingBlockNumber += 1
                #BuildingBlocks[str(Gene)+","+str(Allele)] = AlleleArray
                
                AlleleStrengthArray = []
                AlleleStrengthInitial = PopulationComposition["Gen 0"][2][(NumberOfAlleles-1-Allele),Gene]
                
                for Generation in range(0,NumberOfGenerations):
                    
                    AlleleStrength = PopulationComposition["Gen "+str(Generation)][2][(NumberOfAlleles-1-Allele),Gene]
                    AlleleStrengthArray = np.append(AlleleStrengthArray, AlleleStrength)
                    
                """
                Print the Allele under consideration in both figures
                """
                pp.figure(1)
                figure1  = pp.plot(AlleleStrengthArray, label = str([Gene,Allele,AlleleStrengthInitial]), color = colors[BuildingBlockNumber])
                pp.legend()
                
                pp.figure(2)
                color = colors[BuildingBlockNumber]
                cmap = ListedColormap(['w',color])

                axes[(NumberOfAlleles-1-Allele),Gene].imshow(AlleleArray, cmap=cmap)

                pp.tight_layout()
                

        pp.figure(1)        
        pp.xlabel("Generation Number")
        pp.ylabel("Relative Fitness containing specific Allele")
        pp.title("Allele Strength for each Generation")
        
    
    
    def ShowPopulationSolutions(PopulationDataframe):
        """
        This method shows the different individuals present for a single generation, as specified by the input
        """

        
        
        
        
        pass
        

    

    
    def ShowPopulationConvergence(PopulationComposition, NumberOfGenerations):
        """
        This method visualizes the allele composition of the dictionary PopulationComposition along the generations, for 0%, 50% and 100% progression of generations
        """
        
        HalfGenerations = int(NumberOfGenerations / 2)
        NumberOfImages = 3
        
        # Gather Allele Compositions
        
        AlleleComposition = {}
        AlleleComposition[0] = PopulationComposition["Gen 0"][1] 
        AlleleComposition[1] = PopulationComposition["Gen "+str(HalfGenerations)][1]
        AlleleComposition[2] = PopulationComposition["Gen "+str(NumberOfGenerations)][1]
        
        # Plot Allele Compositions
        
        
        fig, ((ax1), (ax2), (ax3)) = pp.subplots(1, 3, figsize = (20,20))
        
        axes = {}
        axes[0] = ax1
        axes[1] = ax2 
        axes[2] = ax3

        
        image1 = ax1.imshow(AlleleComposition[0], cmap = 'Greens', vmin = 0) 
        image2 = ax2.imshow(AlleleComposition[1], cmap = 'Greens', vmin = 0) 
        image3 = ax3.imshow(AlleleComposition[2], cmap = 'Greens', vmin = 0) 
    
        for image in range(0,3):
            
            ax = axes[image]
            NumberOfGenerationsPlot = round((NumberOfGenerations / NumberOfImages)* (image+1) * (1/NumberOfGenerations) *100,0)
            ax.set_title(str(NumberOfGenerationsPlot)+" % of Generations")

            
            for allele in range(0,AlleleComposition[0].shape[0]):
                
                for gene in range(0,AlleleComposition[0].shape[1]):
                    
                    text = ax.text(gene,allele,AlleleComposition[image][allele,gene], ha= "center",va = "center",color = "w")



    
    
    def create_plot(population_eval, bc,m2, individual_no):
        fig, axs = pp.subplots(3,1, figsize=(15,20))
        S_max = bc.ix["Smax"]
        N = int(sum(population_eval["Cren Design"][individual_no]['dN']))
        t_pattern = population_eval["Chromosome"][individual_no]
        """
        Crenellated panel lifetime values
        """
        #Figure 1. dN set out against the crack length
        population_eval = population_eval["Cren Design"][individual_no]
        axs[0].plot(population_eval['a'],population_eval['dN'], 'r')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs[0].set_title('Fig 1.a: dN per crack growth step')
        axs[0].set_xlabel('crack length a [mm]')
        axs[0].set_xlim([0,500])
        axs[0].set_ylabel('dN [cycles]')
        axs[0].set_yscale('log')
#        axs[0].axhline(y=0, linewidth=1, color = 'b')
        axs[0].legend()
        axs[0].text(350, 0.1*max(population_eval['dN']),'S = '+str(S_max)+' N' '\n' 'N = '+str(N)+' cycles')
        

        #Figure 2. Thickness profile of the crenellation pattern
        axs[1].plot(t_pattern, 'r', label = "Var")
        axs[1].set_title('Fig 1.b: Thickness profile')
        axs[1].set_xlabel('panel half width [mm]')
        axs[1].set_ylabel('thickness t [mm]')
        axs[1].axhline(y=2, linewidth = 1, color= 'g', label = "Ref")
        axs[1].legend()
        
        #Figure 3. The stress distribution along the crack length a
        axs[2].plot(population_eval['a'],population_eval['beta'], 'g')
#        axs[2].plot(population_eval['a'],population_eval['sigma_eff'], 'r', label = "Eff")
#        axs[2].plot(population_eval['a'],population_eval['sigma_iso'], 'b', label = "Iso")
        axs[2].legend()
        axs[2].set_title('Fig 1.c: stress distribution')
        axs[2].set_xlabel('panel half width [mm]')
        axs[2].set_ylabel('beta factor')
        axs[2].set_xlim([0,500])
        
    
        pp.tight_layout()
        pp.show()

    def create_plot_stress_dist(self ):
        pass
        
    def create_plot_detailed(self,population_eval, bc,m2, t_pattern):
        """
        create the subplot structure
        """
        fig, axs = pp.subplots(3,2, figsize=(15,10))
        
        """
        Load in parameters for the plots
        """
        Kth = m2.ix[0,"Kth"]
        Kmax = m2.ix[0,"Kmax"]
        delta_a = bc.ix["crack step size"]
        a_max = bc.ix["Max crack length"]
        a_0 = bc.ix["Initial crack length"]
        total_a = a_max - a_0
        size = int(total_a / delta_a)
        
        """
        Crenellated panel lifetime values
        """
        #First plot dadN - K
        axs[0,1].plot(population_eval['K'],population_eval['dadN'], 'r')
        axs[0,1].set_title('Fig 1.b: Material data')
        axs[0,1].set_xlabel('K [MPa*SQRT(m)]')
        axs[0,1].set_xscale('log')
        axs[0,1].set_xlim([Kth,Kmax])
        axs[0,1].set_ylabel('da/dN [mm/N]')
        axs[0,1].set_yscale('log')
        
        #Second plot dadN - a
        axs[0,0].plot(population_eval['a'],population_eval['dadN'], 'r', label = "Var")
#        axs[0,0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs[0,0].set_title('Fig 1.a: Predicted crack growth')
        axs[0,0].set_xlabel('crack length a [mm]')
        axs[0,0].set_xlim([0,500])
        axs[0,0].set_ylabel('da/dN [mm/N]')
        axs[0,0].set_yscale('log')
        axs[0,0].legend()
         
        #Second plot dadN - a
        axs[1,0].plot(population_eval['a'],population_eval['dN'], 'r', label = "Var")
#        axs[2,0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs[1,0].set_title('Fig 1.e: dN per crack growth step')
        axs[1,0].set_xlabel('crack length a [mm]')
        axs[1,0].set_xlim([0,500])
        axs[1,0].set_ylabel('dN [cycles]')
        axs[1,0].set_yscale('log')
#        axs[2,0].axhline(y=0, linewidth=1, color = 'b')
        axs[1,0].legend()
        
        #Third plot a - K
#        axs[1,0].plot(population_eval['a'],population_eval['K'], 'r', label = "Var")
##        axs[1,0].plot(K_list_ref,a_list_ref, 'g', label = "Ref")
#        axs[1,0].set_title('Fig 1.d: Calculated K')
#        axs[1,0].set_ylabel('K [MPa*SQRT(m)]')
#        axs[1,0].set_ylim([Kth,Kmax])
#        axs[1,0].set_yscale('log')
#        axs[1,0].set_xlabel('crack length a [mm]')
#        axs[1,0].set_xlim([0,500])
#        axs[1,0].legend()
        
        #Fourth plot - thickness profile 
        axs[2,0].plot(t_pattern['Width'],t_pattern['thickness'], 'r', label = "Var")
        axs[2,0].set_title('Fig 1.c: Thickness profile')
        axs[2,0].set_xlabel('panel half width [mm]')
        axs[2,0].set_ylabel('thickness t [mm]')
        axs[2,0].axhline(y=2, linewidth = 1, color= 'g', label = "Ref")
        
        #Fifth plot - a - N
        f_life = population_eval['N'].iloc[-1]
        axs[1,1].plot(population_eval['N'][0:size],population_eval['a'], 'r', label ="Var")
#        axs[1,1].plot(life_ref,a_list_ref, 'g', label = "Ref")
        axs[1,1].set_title('Fig 1.e: Crack length over cycles N')
        axs[1,1].set_xlim([f_life-0.33*f_life,f_life+0.33*f_life])
        axs[1,1].set_xlabel('number of cycles [N]')
        axs[1,1].set_ylabel('crack length a [mm]')
        axs[1,1].set_ylim([0,500])
        axs[1,1].legend()
        
        pp.tight_layout()
        pp.show()
        
    def crenellation_overview_population(self, population):
        
        for i in range(0,len(population["Chromosome"])):
            pp.plot(population["Chromosome"][i])
        
        pp.title('Crenellation patterns within current population')
        pp.tight_layout()
        pp.show()
        
        
    def fitness_plot(self, population_eval,g):
                
        pp.figure(figsize=(50,3))
        pp.axis([200000,300000,0,0])
        
        for i in range(1,len(population_eval["Fitness"])+1):
            value_y = 0
            pp.plot(population_eval["Fitness"][i], value_y, marker= 'o')
            pp.annotate(i, xy=(population_eval["Fitness"][i],0.002))
                  
            
        pp.title('Fitness values for Generation ='+str(g))

        pp.tight_layout()
        pp.show()

    def fittest_crenellation(convergence_overview, run ):
        
        pp.figure(2,figsize=(60,3))
        pp.plot(convergence_overview["Crenellation Pattern"][49][0])
        
        pp.title('Thickness profile fittest individual for run number '+str(run))
        pp.xlabel('panel half width [mm]')
        pp.ylabel('thickness t [mm]')

    
    def convergence(self, convergence_overview, run,bc):
        
        pp.figure(1,figsize=(50,20))
        pop_size = int(bc.ix["Population size"])
        number_of_runs = bc.ix["number_of_runs"]
        generations = bc.ix["Number of Generations"]
        N = np.ceil(int(max(convergence_overview["Fitness"])))
        Pm = bc.ix["Mutation Rate"]
        pp.plot(convergence_overview["Fitness"][:len(convergence_overview)-1], label = "no. "+str(run))

        if run == number_of_runs:
            pp.axhline(y = 261100, xmin = 0 , xmax = generations, linewidth = 3, color = 'k' )
            pp.title('Fitness convergence for several GA optimizations on reference paper')
            pp.text(generations*0.5, 0.95*N,'Pm = '+str(Pm)+ '\n' 'Population Size = '+str(pop_size)+ '\n' 'Max N = '+str(N)+' cycles')
            pp.xlabel('generations')
            pp.ylabel('Fitness (unscaled)')
        #pp.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))

        pp.legend()
                
        
    def create_plot_single(self,fatigue_results, chromosome, bc, m2):
        fig, axs = pp.subplots(2,1, figsize=(40,40))
        t_ref = bc.ix["Reference thickness"]
        S_max = bc.ix["Smax"]
        N = int(sum(fatigue_results['dN']))
        t_pattern = chromosome
        """
        Crenellated panel lifetime values
        """
        #Figure 1. dN set out against the crack length
        population_eval = fatigue_results
        axs[0].plot(population_eval['a'],population_eval['dadN'], 'r')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs[0].set_title('Fig 1.a: dadN per crack growth step')
        axs[0].set_xlabel('crack length a [mm]')
        axs[0].set_xlim([0,np.size(chromosome)])
        axs[0].set_ylabel('dadN [mm/cycle]')
        axs[0].set_ylim([0.0001,0.1])
        axs[0].set_yscale('log')
        axs[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))


#        axs[0].axhline(y=0, linewidth=1, color = 'b')
        axs[0].legend()
        axs[0].text(np.size(chromosome)*0.8, 0.1*max(population_eval['dadN']),'S = '+str(S_max)+' N' '\n' 'N = '+str(N)+' cycles')
        

        #Figure 2. Thickness profile of the crenellation pattern
        axs[1].plot(t_pattern, 'r', label = "Var")
        axs[1].set_title('Fig 1.b: Thickness profile')
        axs[1].set_xlabel('panel half width [mm]')
        axs[1].set_ylabel('thickness t [mm]')
        axs[1].axhline(y= t_ref, linewidth = 1, color= 'g', label = "Ref")
        axs[1].set_xlim([0,np.size(chromosome)])
        axs[1].legend()
        
        #Figure 3. The stress distribution along the crack length a
#        axs[2].plot(population_eval['a'],population_eval['beta'], 'g')
#        axs[2].plot(population_eval['a'],population_eval['sigma_eff'], 'r', label = "Eff")
#        axs[2].plot(population_eval['a'],population_eval['sigma_iso'], 'b', label = "Iso")
#        axs[2].legend()
#        axs[2].set_title('Fig 1.c: stress distribution')
#        axs[2].set_xlabel('panel half width [mm]')
#        axs[2].set_ylabel('beta factor')
#        axs[2].set_xlim([0,np.size(chromosome)])
        
        pp.tight_layout()
        pp.show()
  
    def create_plot_crack_growth(self,fatigue_results, chromosome, bc, m2):

        fig, axs = pp.subplots(1,1, figsize=(40,40))
        t_ref = bc.ix["Reference thickness"]
        S_max = bc.ix["Smax"]
        N = int(sum(fatigue_results['dN']))
        t_pattern = chromosome
        """
        Crenellated panel lifetime values
        """
        #Figure 1. dN set out against the crack length
        population_eval = fatigue_results
        axs.plot(population_eval['a'],population_eval['dadN'], 'r')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Fig 1.a: dadN per crack growth step')
        axs.set_xlabel('crack length a [mm]')
        axs.set_xlim([0,np.size(chromosome)])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.1])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
        
        
    def plot_reference_data_Uz_cren(self, Uz_crenellation_left, Uz_crenellation_right, Huber_cren_prediction, fatigue_results, chromosome, bc):
        
        fig, axs = pp.subplots(1,1, figsize=(40,40))

        """
        Crenellated panel lifetime values
        """
        #Figure 1. Uz reference data left
        axs.plot(Uz_crenellation_left['Crack length'],Uz_crenellation_left['dadN'], 'r--', label ='Uz et al test (left)')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Crack growth rate for crenellated plate loaded with 50 MPa')
        axs.set_xlabel('crack length a [mm]')
        axs.set_xlim([0,300])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.01])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
        #Figure 2. Uz reference data right
        axs.plot(Uz_crenellation_right['Crack length'],Uz_crenellation_right['dadN'] ,'b--', label ='Uz et al test (right)')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Crack growth rate for crenellated plate loaded with 50 MPa')
        axs.set_xlabel('crack length a [mm]')
        axs.set_xlim([0,300])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.01])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
                #Figure 2. Uz reference data right
        axs.plot(Huber_cren_prediction['Crack length'],Huber_cren_prediction['dadN'],'g', linewidth = 2, label = 'Huber et al Prediction FEM')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Crack growth rate for crenellated plate loaded with 50 MPa')
        axs.set_xlabel('crack length a [mm]')
        axs.set_xlim([0,300])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.01])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
        population_eval = fatigue_results
        axs.plot(population_eval['a'],population_eval['dadN'], 'y', linewidth = 2, label = 'Analytical model')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Crack growth rate for crenellated plate loaded with 50 MPa')
        axs.set_xlabel('crack length a [mm]')
        axs.set_xlim([0,300])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.01])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
        axs2 = axs.twinx()
        #Figure 1. dN set out against the crack length
        t_pattern = chromosome
        axs2.plot(t_pattern)
        axs2.set_ylim([0,50])
        x = np.arange(0,300,1)
        axs2.fill_between(x, 0.0001, t_pattern)
        axs2.set_ylabel("plate thickness [mm]")
        
        
        
        axs.legend()
        
        pp.tight_layout()
        pp.show()

        
    def plot_single_pattern(self, fatigue_results):
        
        fig, axs = pp.subplots(1,1, figsize=(40,40))

        population_eval = fatigue_results
        axs.plot(population_eval['a'],population_eval['dadN'], 'y', linewidth = 2, label = 'Analytical model')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Crack growth rate vs SIF for flat plate loaded with 50 MPa')
        axs.set_xlabel('Stress Intensity Factor K [MPa/m^0.5]')
        axs.set_xlim([0,150])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.01])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
        
    def plot_reference_data_Uz_uniform(self, Uz_uniform_right, fatigue_results, chromosome, bc):
        
        fig, axs = pp.subplots(1,1, figsize=(40,40))

        #Figure 1. Uz reference data right
        axs.plot(Uz_uniform_right['Crack length'],Uz_uniform_right['dadN'] ,'b--', label ='Uz et al test (right)')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Crack growth rate for crenellated plate loaded with 50 MPa')
        axs.set_xlabel('crack length a [mm]')
        axs.set_xlim([0,300])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.1])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
        
        population_eval = fatigue_results
        axs.plot(population_eval['a'],population_eval['dadN'], 'y', linewidth = 2, label = 'Analytical model')
#        axs[0].plot(a_list_ref,dadN_list_ref, 'g', label = "Ref")
        axs.set_title('Crack growth rate for crenellated plate loaded with 50 MPa')
        axs.set_xlabel('crack length a [mm]')
        axs.set_xlim([0,300])
        axs.set_ylabel('dadN [mm/cycle]')
        axs.set_ylim([0.0001,0.1])
        axs.set_yscale('log')
        axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        
        axs2 = axs.twinx()
        #Figure 1. dN set out against the crack length
        t_pattern = chromosome
        axs2.plot(t_pattern)
        axs2.set_ylim([0,50])
        x = np.arange(0,300,1)
        axs2.fill_between(x, 0.0001, t_pattern)
        axs2.set_ylabel("plate thickness [mm]")
        
        
        axs.legend()
        
        pp.tight_layout()
        pp.show()
        
    def brute_force_comparison(self, population_eval_sorted):
        
        
        
        pass
        
        
    
    def plot_histogram(self, population_eval_sorted_5cont_8thick):
        
        pp.hist(population_eval_sorted_5cont_8thick.Fitness[:4603], bins = 400)   
        pp.xlabel("Fitness level (fatigue life in cycles N)")
        pp.ylabel("Number of Unique Crenellation Patterns")
        pp.title("Brute force calculation of fitness levels for feasible crenellation patterns")
        
#    
#    
#    def animate_store(self):
#        
#    image = pp.imshow(fitness_plot, animated=True)    
#
#    fig = pp.figure()
#    ani = animation.FuncAnimation(fig, fitness_plot, g)
#        
#        
#        
        
        