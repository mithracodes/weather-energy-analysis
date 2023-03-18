import pandas as pd
import helper_functions as helper

# import csv files to be used as dataframes
adelaide_df = pd.read_csv("/home/AdelaideCSV/adelaide_clean.csv")
brisbane_df = pd.read_csv("/home/BrisbaneCSV/brisbane_clean.csv")
melbourne_df = pd.read_csv("/home/MelbourneCSV/melbourne_clean.csv")
sydney_df = pd.read_csv("/home/SydneyCSV/sydney_clean.csv")

melbourne_iqr_df = pd.read_csv("/home/MelbourneCSV/melb_iqr.csv")
sydney_iqr_df = pd.read_csv("/home/SydneyCSV/syd_iqr.csv")
brisbane_iqr_df = pd.read_csv("/home/BrisbaneCSV/bris_iqr.csv")
adelaide_iqr_df = pd.read_csv("/home/AdelaideCSV/adel_iqr.csv")


# non linear regression for non linear features
helper.mintemp_nonlinear(sydney_df, 'Sydney')
helper.maxtemp_nonlinear(sydney_df, 'Sydney')
helper.evaporation_nonlinear(sydney_iqr_df, 'Sydney')

helper.mintemp_nonlinear(melbourne_df, 'Melbourne')
helper.maxtemp_nonlinear(melbourne_df, 'Melbourne')
helper.evaporation_nonlinear(melbourne_iqr_df, 'Melbourne')

helper.mintemp_nonlinear(brisbane_df, 'Brisbane')
helper.maxtemp_nonlinear(brisbane_df, 'Brisbane')
helper.evaporation_nonlinear(brisbane_iqr_df, 'Brisbane')

helper.mintemp_nonlinear(adelaide_df, 'Adelaide')
helper.maxtemp_nonlinear(adelaide_df, 'Adelaide')



