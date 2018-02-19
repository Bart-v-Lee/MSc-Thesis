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
        
    def RetrieveMaterial():
        """
        """
        
        
        
        
        
        pass
    
    def RetrieveBoundaryConditions():
        """
        """
        
        delta_x = 
        W = 
        N_pop = 
        t_dict = 
        
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

        """
        
        
        
        
        
        pass