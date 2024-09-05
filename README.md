# PRE_2024_Paper_Figures
 This code runs the analysis and generates figures for the paper  "Wind-driven variability in the prereversal enhancement: Climatologies observed by ICON" by Harding et al. (2024), JGR Space Physics
 
 - PRE_2024_Paper_Figures.ipynb: This notebook generates the figures. Start here.
 - TIEGCM_dynamo_modeling.ipynb: This notebook runs the modeling studies and saves a file "tiegcm_2024_v05.nc" which is used in the notebook above. This takes several hours to run at full resolution, so a precomputed version of this file can be downloaded instead of recreating it yourself.
 - dynamo.py: This Python module contains the standalone dynamo solver described in the paper.
