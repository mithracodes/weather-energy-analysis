import pandas as pd
import helper_functions as helper

# gathered by visually scanning the price_demand_data.csv
region1 = 'NSW1'
region2 = 'VIC1'
region3 = 'QLD1'
region4 = 'SA1'

# read in data
price_df = pd.read_csv("price_demand_data.csv")
adelaide_df = pd.read_csv("weather_adelaide.csv")
brisbane_df = pd.read_csv("weather_brisbane.csv")
melbourne_df = pd.read_csv("weather_melbourne.csv")
sydney_df = pd.read_csv("weather_sydney.csv")

# condense energy data into 30 minute segments
nsw_df = helper.condense_demand_data(region1, price_df)
vic_df = helper.condense_demand_data(region2, price_df)
qld_df = helper.condense_demand_data(region3, price_df)
sa_df = helper.condense_demand_data(region4, price_df)

# calculate daily energy demand averages
vic_clean_df = helper.calc_averages(vic_df) 
nsw_clean_df = helper.calc_averages(nsw_df) 
sa_clean_df = helper.calc_averages(sa_df) 
qld_clean_df = helper.calc_averages(qld_df) 

# note 1: the dataset finishes at the start of 2022/03/18,
# as the there is only 1 recorded entry, it is ignored
# Additionally, the dataset starts at 00:30:00 so it is missing 
# one entry

# note 2: as of the 2021/10/01, demand date was collected every 5 minutes
# as opposed to the previous 30 minute intervals. These 5 minute intervals have 
# been filtered out to retain consistency in averages

# remove columns with completely missing entries
adelaide_df = adelaide_df.drop("Evaporation (mm)", axis=1)
adelaide_df = adelaide_df.drop("Sunshine (hours)", axis=1)
brisbane_df = brisbane_df.drop("Evaporation (mm)", axis=1)


# combine energy data with weather data
sydney_df = helper.combine_and_clean(nsw_clean_df, sydney_df)
melbourne_df = helper.combine_and_clean(vic_clean_df, melbourne_df)
brisbane_df = helper.combine_and_clean(qld_clean_df, brisbane_df)
adelaide_df = helper.combine_and_clean(sa_clean_df, adelaide_df)

# export to csv to be used by other files
sydney_df.to_csv('/home/SydneyCSV/sydney_clean.csv', index = False)
melbourne_df.to_csv('/home/MelbourneCSV/melbourne_clean.csv', index = False)
brisbane_df.to_csv('/home/BrisbaneCSV/brisbane_clean.csv', index = False)
adelaide_df.to_csv('/home/AdelaideCSV/adelaide_clean.csv', index = False)

# remove outliers using upper and lower bounds (IQR)
sydney_iqr_df = helper.outliers_iqr(sydney_df)
melbourne_iqr_df = helper.outliers_iqr(melbourne_df)
brisbane_iqr_df = helper.outliers_iqr(brisbane_df)
adelaide_iqr_df = helper.outliers_iqr(adelaide_df)

melbourne_iqr_df.to_csv('/home/MelbourneCSV/melb_iqr.csv', index = False)
sydney_iqr_df.to_csv('/home/SydneyCSV/syd_iqr.csv', index = False)
brisbane_iqr_df.to_csv('/home/BrisbaneCSV/bris_iqr.csv', index = False)
adelaide_iqr_df.to_csv('/home/AdelaideCSV/adel_iqr.csv', index = False)

# remove outliers using z-scores
sydney_zscore_df = helper.outliers_zscore(sydney_df)
melbourne_zscore_df = helper.outliers_zscore(melbourne_df)
brisbane_zscore_df = helper.outliers_zscore(brisbane_df)
adelaide_zscore_df = helper.outliers_zscore(adelaide_df)

melbourne_zscore_df.to_csv('/home/MelbourneCSV/melb_zscore.csv', index = False)
sydney_zscore_df.to_csv('/home/SydneyCSV/syd_zscore.csv', index = False)
brisbane_zscore_df.to_csv('/home/BrisbaneCSV/bris_zscore.csv', index = False)
adelaide_zscore_df.to_csv('/home/AdelaideCSV/adel_zscore.csv', index = False)

