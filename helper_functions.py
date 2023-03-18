import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesClassifier

################################################################################

def condense_demand_data(region, demand_df):
    rows = []
    for index, row in demand_df.iterrows():
        # find every row for a certain region (NSW,VIC,SA,QLD)
        if (row[0]==region):
            # only use data for 30 minute intervals
            if((row[1][14:16]=='30') or (row[1][14:16]=='00')):
                rows.append({
                    "Date" : row[1][:10],
                    "Demand (MW)" : row[2],
                })
    df = pd.DataFrame(rows)
    return df

################################################################################

def calc_averages(df):
    rows = []
    curr_date = df['Date'][0]
    demand_sum = 0
    entry_count = 0
    #loop through, combining data from the same day
    for index, row in df.iterrows():
        if(row['Date']==curr_date):
            demand_sum = demand_sum + int(row[1])
            entry_count+=1
        else:
            # once a different date is reached, calculate the average for previous date
            demand_sum = demand_sum/entry_count
            rows.append({
                "Date" : curr_date,
                "Average Demand (MW)" : float('{:.2f}'.format(demand_sum))
            })
            # set the new date as the current and reset other variables
            curr_date = row['Date']
            demand_sum = int(row[1])
            entry_count = 0

    clean_df = pd.DataFrame(rows)
    return clean_df

################################################################################

def combine_and_clean(clean_df, weather_df):
    averages = clean_df['Average Demand (MW)']
    # add averages to weather data
    weather_df = weather_df.join(averages)
    # remove features determined by feature selection 
    weather_df = weather_df.drop(['Direction of maximum wind gust ', 
    'Speed of maximum wind gust (km/h)', 'Time of maximum wind gust', 
    '9am Temperature (°C)', '9am relative humidity (%)', '9am cloud amount (oktas)', 
    '9am wind direction', '9am wind speed (km/h)', '9am MSL pressure (hPa)', 
    '3pm Temperature (°C)', '3pm relative humidity (%)', '3pm cloud amount (oktas)', 
    '3pm wind direction', '3pm wind speed (km/h)', '3pm MSL pressure (hPa)'], axis = 1)
    weather_df = weather_df.set_index('Date')
    # ensure a consistent set of dates (2021/02/01 - 2022/02/01)
    weather_df = weather_df.drop(weather_df.tail(58).index)
    # fill in values that are missing via linear interpolation
    weather_df = weather_df.interpolate()
    return weather_df

################################################################################

def outliers_iqr(weather_df): 
    # determine 1st and 3rd quartiles
    Q1 = weather_df.quantile(0.25)
    Q3 = weather_df.quantile(0.75)
    # determine iqr
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    # remove all values outside of the upper and lower bounds
    iqr_df = weather_df[~((weather_df < lower_bound) | (weather_df > upper_bound)).any(axis=1)]
    return iqr_df

################################################################################

def outliers_zscore(weather_df):
    # calculate z scores
    z_scores = stats.zscore(weather_df)
    # take the absolute values as direction is irrelavant
    abs_z_scores = np.abs(z_scores)
    # create list of values that are below set threshold
    scores = (abs_z_scores < 3).all(axis=1)
    # make this list the new dataframe, thereby removing outliers
    zscore_df = weather_df[scores]
    return zscore_df
    # Above code fragment from the following source: 
    # https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe
################################################################################

def rainfall_plot(city_df, city):
    # extract column and reshape into 2D array
    X = city_df.iloc[:,2].values.reshape(-1, 1)
    # extract the energy demand column (final column)
    Y = city_df.iloc[:,-1].values.reshape(-1, 1)
    regressor = LinearRegression()
    # fit data to linear model
    regressor.fit(X, Y)
    # create list of predicted values from X
    y_pred = regressor.predict(X)
    # create plot with line of linear regression
    plt.scatter(X, Y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('Linear Regression: Rainfall and Energy Demand')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Energy Demand (MW)')
    plt.show()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'Rainfall.png') 
    plt.clf()

    # create residual plot
    residual = Y-y_pred
    sns.residplot(x=y_pred, y=residual, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    plt.ylabel('Residual')
    plt.xlabel('Predicted Energy Demand (MW)')
    plt.title(label='Residual Plot: Rainfall')
    plt.tight_layout()
    plt.savefig('ResidualsRainfall.png')
    plt.clf()

################################################################################

# see above for commenting
def evaporation_plot(city_df, city):
    X = city_df.iloc[:,3].values.reshape(-1, 1)
    Y = city_df.iloc[:,-1].values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(X, Y)
    y_pred = regressor.predict(X)
    plt.scatter(X, Y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('Linear Regression: Evaporation and Energy Demand')
    plt.xlabel('Evaporation (mm)')
    plt.ylabel('Energy Demand (MW)')
    plt.show()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'Evaporation.png')
    plt.clf()

    residual = Y-y_pred
    sns.residplot(x=y_pred, y=residual, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    plt.ylabel('Residual')
    plt.xlabel('Predicted Energy Demand (MW)')
    plt.title(label='Residual Plot: Evaporation')
    plt.tight_layout()
    plt.savefig('ResidualsEvaporation.png')
    plt.clf()
    
################################################################################

# see above for commenting
def mintemp_plot(city_df, city):
    X = city_df.iloc[:,0].values.reshape(-1, 1)
    Y = city_df.iloc[:,-1].values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(X, Y)
    y_pred = regressor.predict(X)
    plt.scatter(X, Y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('Linear Regression: Min Temp and Energy Demand')
    plt.xlabel('Minimum temperature (°C)')
    plt.ylabel('Energy Demand (MW)')
    plt.show()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'MinTemp.png') 
    plt.clf()

    residual = Y-y_pred
    sns.residplot(x=y_pred, y=residual, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    plt.ylabel('Residual')
    plt.xlabel('Predicted Energy Demand (MW)')
    plt.title(label='Residual Plot: MinTemp')
    plt.tight_layout()
    plt.savefig('ResidualsMinTemp.png')
    plt.clf()
    
################################################################################

# see above for commenting
def maxtemp_plot(city_df, city):
    X = city_df.iloc[:,1].values.reshape(-1, 1)
    Y = city_df.iloc[:,-1].values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(X, Y)
    y_pred = regressor.predict(X)
    plt.scatter(X, Y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('Linear Regression: Max Temp and Energy Demand')
    plt.xlabel('Maximum temperature (°C)')
    plt.ylabel('Energy Demand (MW)')
    plt.show()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'MaxTemp.png') 
    plt.clf()

    residual = Y-y_pred
    sns.residplot(x=y_pred, y=residual, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    plt.ylabel('Residual')
    plt.xlabel('Predicted Energy Demand (MW)')
    plt.title(label='Residual Plot: MaxTemp')
    plt.tight_layout()
    plt.savefig('ResidualsMaxTemp.png')
    plt.clf()
    
################################################################################

# see above for commenting
def sunshinehours_plot(city_df, city):
    if (city == 'Brisbane'):
        X = city_df.iloc[:,3].values.reshape(-1, 1)
    else:
        X = city_df.iloc[:,4].values.reshape(-1, 1)
    Y = city_df.iloc[:,-1].values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(X, Y)
    y_pred = regressor.predict(X)
    plt.scatter(X, Y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('Linear Regression: Sunshine and Energy Demand')
    plt.xlabel('Sunshine (hours)')
    plt.ylabel('Energy Demand (MW)')
    plt.show()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'SunshineHours.png') 
    plt.clf()

    residual = Y-y_pred
    sns.residplot(x=y_pred, y=residual, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    plt.ylabel('Residual')
    plt.xlabel('Predicted Energy Demand (MW)')
    plt.title(label='Residual Plot: Sunshine')
    plt.tight_layout()
    plt.savefig('ResidualsSunshine.png')
    plt.clf()

################################################################################

def mintemp_outliers(city_df):
    # calculate pearsons coefficient correlations for the dataframe
    correlation = city_df.corr()
    # return only the needed features
    return round(correlation.loc['Average Demand (MW)', 'Minimum temperature (°C)'],3)

################################################################################

# see above for commenting
def maxtemp_outliers(city_df):
    correlation = city_df.corr()
    return round(correlation.loc['Average Demand (MW)', 'Maximum temperature (°C)'],3)

################################################################################

# see above for commenting
def rainfall_outliers(city_df):
    correlation = city_df.corr()
    return round(correlation.loc['Average Demand (MW)', 'Rainfall (mm)'],3)

################################################################################

# see above for commenting
def evaporation_outliers(city_df, city):
    # account for missing features
    if (city == 'Adelaide') or (city == 'Brisbane'):
        return 'N/A'
    correlation = city_df.corr()
    return round(correlation.loc['Average Demand (MW)', 'Evaporation (mm)'],3)

################################################################################

# see above for commenting
def sunshine_outliers(city_df, city):
    if (city=='Adelaide'):
        return 'N/A'
    correlation = city_df.corr()
    return round(correlation.loc['Average Demand (MW)', 'Sunshine (hours)'],3)


################################################################################

# create table for each method of removing outliers
def create_outlier_table(city_df, city_iqr_df, city_zscore_df, city):
    new_table = []
    new_table.append({
        "Method" : 'None',
        "Rainfall" : rainfall_outliers(city_df),
        "Evaporation" : evaporation_outliers(city_df, city),
        "Sunshine" : sunshine_outliers(city_df, city),
        "MinTemp" : mintemp_outliers(city_df),
        "MaxTemp" : maxtemp_outliers(city_df)
    })
    new_table.append({
        "Method" : 'IQR',
        "Rainfall" : rainfall_outliers(city_iqr_df),
        "Evaporation" : evaporation_outliers(city_iqr_df, city),
        "Sunshine" : sunshine_outliers(city_iqr_df, city),
        "MinTemp" : mintemp_outliers(city_iqr_df),
        "MaxTemp" : maxtemp_outliers(city_iqr_df)
    })

    new_table.append({
        "Method" : 'Z-Score',
        "Rainfall" : rainfall_outliers(city_zscore_df),
        "Evaporation" : evaporation_outliers(city_zscore_df, city),
        "Sunshine" : sunshine_outliers(city_zscore_df, city),
        "MinTemp" : mintemp_outliers(city_zscore_df),
        "MaxTemp" : maxtemp_outliers(city_zscore_df)
    })

    table_df = pd.DataFrame(new_table)
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.set_title("Pearsons correlation for " + city + " weather against energy demand")
    ax.table(cellText=table_df.values, colLabels=table_df.columns, loc='center')
    fig.tight_layout()
    plt.savefig(r'/home/Tables/Outliers_' + city + '_table.png')
    plt.clf()

################################################################################

def heatmap(city_df, city):
    X = city_df.iloc[:,0:5]
    y = city_df.iloc[:,-1]
    # get correlations of each features in dataset
    corrmat = city_df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    # plot heat map
    g=sns.heatmap(city_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    # Above code fragment from the following source:
    # https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

    title = 'Heatmap for weather conditions - ' + city
    plt.title(title, fontsize=40, loc='center')
    plt.tight_layout()
    plt.savefig(r'/home/' + city + 'Plots/' + city + '_heatmap.png')
    # return figure size to default
    plt.figure(figsize=(6.4,4.8))
    plt.clf()

################################################################################

def mintemp_nonlinear(city_df, city):
    # create training split at 80%
    training_data = city_df.sample(frac=0.8, random_state=25)
    x_train = training_data.iloc[:,0]
    y_train = training_data.iloc[:,-1]
    
    # create testing split from the remaining 20%
    testing_data = city_df.drop(training_data.index)
    x_test = testing_data.iloc[:,0]
    y_test = testing_data.iloc[:,-1]

    # create model
    mymodel = np.poly1d(np.polyfit(x_train, y_train, 3))
    # Above code fragment from the following source:
    # https://www.w3schools.com/python/python_ml_train_test.asp

    # fit model onto plot
    myline = np.linspace(min(x_train), max(x_train), 100)
    plt.scatter(x_train, y_train, color ='red')
    plt.plot(myline, mymodel(myline), color='blue', linewidth=2)
    title = 'Non-Linear Regression ' + city + ': MinTemp'
    plt.title(title)
    plt.xlabel('Minimum Temperature (°C)')
    plt.ylabel('Energy Demand (MW)')
    plt.tight_layout()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'MinTempRegression.png')
    plt.clf()

    # predicted energy demand from model
    y_pred = mymodel(x_test)
    # find difference from actual energy demand
    residuals = y_test-y_pred
    # plot residuals
    sns.residplot(x=y_pred, y=residuals, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    title = 'Non-linear residual plot ' + city + ': MinTemp'
    plt.title(title)
    plt.xlabel('Energy Demand (MW)')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'MinTempRegressionResidual.png') 
    plt.clf()   

################################################################################

# see above for commenting
def maxtemp_nonlinear(city_df, city):
    training_data = city_df.sample(frac=0.8, random_state=25)
    x_train = training_data.iloc[:,1]
    y_train = training_data.iloc[:,-1]

    testing_data = city_df.drop(training_data.index)
    x_test = testing_data.iloc[:,1]
    y_test = testing_data.iloc[:,-1]

    mymodel = np.poly1d(np.polyfit(x_train, y_train, 3))

    myline = np.linspace(min(x_train), max(x_train), 100)
    plt.scatter(x_train, y_train, color ='red')
    plt.plot(myline, mymodel(myline), color='blue', linewidth=2)
    title = 'Non-Linear Regression ' + city + ': MaxTemp'
    plt.title(title)
    plt.xlabel('Maximum Temperature (°C)')
    plt.ylabel('Energy Demand (MW)')
    plt.tight_layout()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'MaxTempRegression.png')
    plt.clf()

    trainr2 = r2_score(y_train, mymodel(x_train))
    testr2 = r2_score(y_test, mymodel(x_test))
    y_pred = mymodel(x_test)
    residuals = y_test-y_pred
    sns.residplot(x=y_pred, y=residuals, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    title = 'Non-linear residual plot ' + city + ': MaxTemp'
    plt.title(title)
    plt.xlabel('Energy Demand (MW)')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'MaxTempRegressionResidual.png') 
    plt.clf()   

################################################################################

# see above for commenting
def evaporation_nonlinear(city_df, city):
    training_data = city_df.sample(frac=0.8, random_state=25)
    x_train = training_data.iloc[:,3]
    y_train = training_data.iloc[:,-1]

    testing_data = city_df.drop(training_data.index)
    x_test = testing_data.iloc[:,3]
    y_test = testing_data.iloc[:,-1]

    mymodel = np.poly1d(np.polyfit(x_train, y_train, 3))

    myline = np.linspace(min(x_train), max(x_train), 100)
    plt.scatter(x_train, y_train, color ='red')
    plt.plot(myline, mymodel(myline), color='blue', linewidth=2)
    title = 'Non-Linear Regression ' + city + ': Evaporation'
    plt.title(title)
    plt.xlabel('Evaporation (mm)')
    plt.ylabel('Energy Demand (MW)')
    plt.tight_layout()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'EvaporationRegression.png')
    plt.clf()

    trainr2 = r2_score(y_train, mymodel(x_train))
    testr2 = r2_score(y_test, mymodel(x_test))
    y_pred = mymodel(x_test)
    residuals = y_test-y_pred
    sns.residplot(x=y_pred, y=residuals, lowess=True, scatter_kws={"color": "red"}, line_kws={"color": "blue"})  #order=3 for non-linear
    title = 'Non-linear residual plot ' + city + ': Evaporation'
    plt.title(title)
    plt.xlabel('Energy Demand (MW)')
    plt.ylabel('Residual')
    plt.tight_layout()
    'MinTemp.png'
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'EvaporationRegressionResidual.png') 
    plt.clf() 

################################################################################

def feature_importances(city_df, city): 
    # account for missing features
    if city == 'Brisbane':
        X = city_df.iloc[:,0:4]
    if city == 'Adelaide':
        X = city_df.iloc[:,0:3]
    if (city == 'Melbourne') or (city == 'Sydney'):
        X = city_df.iloc[:,0:5]
    y = city_df.iloc[:,-1]
    # ensure data is in the correct format
    y=y.astype('int')

    # create model using decision tree
    model = ExtraTreesClassifier()
    model.fit(X,y)
    # plot features 
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.plot(kind='barh')
    plt.tight_layout()
    plt.savefig(r'/home/' + city + 'Plots/' + city + 'Feature_importances' + city + '.png')
    plt.clf()
    # Above code from the following source:
    # https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

################################################################################

