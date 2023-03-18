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

melbourne_zscore_df = pd.read_csv("/home/MelbourneCSV/melb_zscore.csv")
sydney_zscore_df = pd.read_csv("/home/SydneyCSV/syd_zscore.csv")
brisbane_zscore_df = pd.read_csv("/home/BrisbaneCSV/bris_zscore.csv")
adelaide_zscore_df = pd.read_csv("/home/AdelaideCSV/adel_zscore.csv")


# feature importance graphs
helper.feature_importances(sydney_df, 'Sydney')
helper.feature_importances(brisbane_df, 'Brisbane')
helper.feature_importances(adelaide_df, 'Adelaide')
helper.feature_importances(melbourne_df, 'Melbourne')


# heatmap plots
helper.heatmap(brisbane_df, 'Brisbane')
helper.heatmap(melbourne_df, 'Melbourne')
helper.heatmap(sydney_df, 'Sydney')
helper.heatmap(adelaide_df, 'Adelaide')


# table for correlation with outliers, and without (two methods)
helper.create_outlier_table(melbourne_df, melbourne_iqr_df, melbourne_zscore_df, 'Melbourne')
helper.create_outlier_table(sydney_df, sydney_iqr_df, sydney_zscore_df, 'Sydney')
helper.create_outlier_table(brisbane_df, brisbane_iqr_df, brisbane_zscore_df, 'Brisbane')
helper.create_outlier_table(adelaide_df, adelaide_iqr_df, adelaide_zscore_df, 'Adelaide')


# individual plots for each feature, for each city
helper.rainfall_plot(melbourne_df, 'Melbourne')
helper.mintemp_plot(melbourne_df, 'Melbourne')
helper.maxtemp_plot(melbourne_df, 'Melbourne')
helper.sunshinehours_plot(melbourne_df, 'Melbourne')
helper.evaporation_plot(melbourne_df, 'Melbourne')

helper.rainfall_plot(sydney_df, 'Sydney')
helper.mintemp_plot(sydney_df, 'Sydney')
helper.maxtemp_plot(sydney_df, 'Sydney')
helper.sunshinehours_plot(sydney_df, 'Sydney')
helper.evaporation_plot(sydney_df, 'Sydney')

helper.rainfall_plot(brisbane_df, 'Brisbane')
helper.mintemp_plot(brisbane_df, 'Brisbane')
helper.maxtemp_plot(brisbane_df, 'Brisbane')
helper.sunshinehours_plot(brisbane_df, 'Brisbane')

helper.rainfall_plot(adelaide_df, 'Adelaide')
helper.mintemp_plot(adelaide_df, 'Adelaide')
helper.maxtemp_plot(adelaide_df, 'Adelaide')
