# weather-energy-analysis

Other Contributors: *William Walker* and *Jack Olivier*

## Overview

The aim of this project is to investigate the extent by which various weather-related parameters such as Rainfall (mm), Sunshine (hours), Evaporation (mm), Minimum Temperature (°C) and Maximum Temperature (°C) affect the energy demand of a particular city, and how this varies between the 4 cities: Adelaide, Brisbane, Melbourne and Sydney. 

## File Structure and Purpose

This repository contains four Python files, each with a specific purpose:

- `main.py`: This file is responsible for most of the pre-processing and has the longest runtime. It reads in the data, cleans and processes it, and prepares it for modeling.
* `plotting.py`: This file is responsible for exporting most of the plots that were created during the analysis. It uses the processed data from main.py to generate visualizations for exploratory data analysis and model evaluation.
+ `modelling.py`: This file is responsible for running the modelling functions for non-linear regression. It takes in the processed data from main.py and trains several non-linear regression models using various algorithms.
- `helper_functions.py`: This file holds all of the functions used in the previous three Python files. It is imported by each file to utilize the common functions.

## To run the program

The three core files should be run in the following order:

1. `main.py`: Run this file to process the data and prepare it for modeling. This will generate several output files that will be used by plotting.py and modelling.py.

2. `plotting.py`: Run this file to generate visualizations of the processed data and model outputs. This will export several image files to the output/ directory.

3. `modelling.py`: Run this file to train non-linear regression models using the processed data. This will output the trained model objects to the output/ directory.

Note that `helper_functions.py` does not need to be run as it only contains functions that are imported by the other files.

## Datasets

This investigation utilized five datasets, all of which were in Excel file format:

- Four datasets consisting of daily weather observations from *1/3/2021 to 31/3/2022*, recorded by the *Australian Government Bureau of Meteorology website* for four weather stations in *VIC, NSW, QLD, and SA*, respectively.

* A dataset composed of *Aggregated Price and Demand* data recorded from *1/2/2021 to 18/3/2022* for the states *NSW, QLD, SA, and VIC*, respectively.

The weather-related parameters used as explanatory variables in this investigation were:

1. Rainfall (mm)
2. Sunshine (hours)
3. Evaporation (mm)
4. Minimum Temperature (°C)
5. Maximum Temperature (°C)

The daily weather observations datasets contained these parameters along with the dates on which they were recorded. The Aggregated Price and Demand dataset contained the total energy demand recorded every 30 minutes, along with a boolean value indicating whether there was a price surge during those 30 minutes.

The dates on which these parameters were recorded were used to link the Daily Weather Observations and the Aggregated Price and Demand data. Additionally, the total energy demand recorded every 30 minutes in the Aggregated Price and Demand dataset was averaged to reflect the mean energy demand of the entire day for a particular date.

*** Note: This is my team's *Assignment 2 of COMP20008 Elements of Data Processing in Sem 1 2022*. ***
