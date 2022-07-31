from multiprocessing import dummy
try:

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import time
    import warnings
    warnings.filterwarnings('ignore')
    #%matplotlib inline
    from sklearn import ensemble
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
except ImportError as error:
    raise error


class pre_process:
    """
    This Class is used for loading and preprocessing of data required by ML model later
    """

    def __init__(self) -> None:
        """
        --> DocString here <--
        """
        pass

    def load_data(data_path):
        """
        This function reads the csv file and loads it into a DataFrame.

        Argument:
        data_path: Path where data file is present

        Returns:
        df: pandas DataFrame that has the data.
        
        """
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return df
        else:
            print("File not found at: ", path)

        
    def rename_columns(df):
        """
        This function renames the columns to make it more readable
        
        Argument:
        df: dataframe
        
        """
        df = df.rename(columns={'dteday':'date','yr':'year','hr': 'hour','mnth':'month','weathersit':'weather',
                       'hum':'humidity','cnt':'count'})
        return df

    def find_dtypes(df):
        """
        This function prints the data type.
        
        Argument:
        df: DataFrame
        
        """
        print('Data types of the columns in dataset are: \n',df.dtypes)
     

    def convert_dtypes(df):
        """
        Convert Date: from object to date
                season,year,month,holiday,weekday,workingday,weather from int64 to category
        
        Argument:
        df: DataFrame
        
        """
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        column_names = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weather']
        for column in column_names:
            df[column] = df[column].astype('category')
        
        print('Df with change data type: ',df.dtypes)
        
        return df

        
    def check_missing_value(df):
        """
        This function check the missing/null value in the data
        
        Argument:
        df: DataFrame

        """
        print('Missing values in the dataframe: \n')
        print(df.isnull().sum())


    def map_col_values(df):
        """
        For better clarity this function maps the values(numbers) in season, month, weekday and weather columns to corresponding
        description
        
        Argument:
        df: DataFrame

        """
        # Mapping season
        df['season'] = df['season'].map({1: 'Spring', 2: 'Summer',
                                           3: 'Fall', 4: 'Winter'})
        # mapping month
        df['month'] = df['month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'April',
                                       5:'May', 6:'June', 7:'July', 8:'Aug', 
                                       9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'})
        # mapping weekday
        df['weekday'] = df['weekday'].map({1: 'Mon', 2: 'Tues', 3: 'Wed', 4: 'Thurs',
                                       5:'Fri', 6:'Sat', 0:'Sun'})
        # mapping weather
        df['weather'] = df['weather'].map({1: 'Clear or Partly Cloudy', 2: 'Misty and Cloudy',
                                           3: 'Light Snow or Rain or Scattered Clouds', 4: 'Heavy Rain or Snow'})

        # mapping working day
        df['workingday'] = df['workingday'].map({1: 'working day', 0: 'holiday'})
        
        return df

    def outlier_analysis(df):
        """
        This function plots various boxplots to analyse the outlier
        
        Argument:
        df: DataFrame

        """
        plt.rcParams['axes.labelsize'] = 10
        #plt.rcParams['axes.title'] = 20
        fig, axes = plt.subplots(nrows=4, ncols=2)
        fig.set_size_inches(30, 25)
        plt.subplots_adjust(hspace = 0.5)
        sns.boxplot(data=df, y="count",x="weather", orient="v", ax=axes[0][0])
        sns.boxplot(data=df, y="count", x="season", orient="v", ax=axes[0][1])
        sns.boxplot(data=df, y="count", x="month", orient="v", ax=axes[1][0])
        sns.boxplot(data=df, y="count", x="workingday", orient="v", ax=axes[1][1])
        sns.boxplot(data=df, y="count", x="hour", orient="v", ax=axes[2][0])
        sns.boxplot(data=df, y="count", orient="v", ax=axes[2][1])
        sns.boxplot(data=df, y="count", x="weekday",orient="v", ax=axes[3][0])

        axes[0][0].set(ylabel="count", 
                   title="Box Plot On count across weather")
        axes[0][1].set(
        xlabel="Season",
        ylabel="Count", 
        title="Box Plot On Count Season on Season"
        )
        axes[1][0].set(
        xlabel="Month",
        ylabel="Count",
        title="Box Plot across Months",
        )
        axes[1][1].set(
        xlabel="Working Day",
        ylabel="Count",
        title="Box Plot On Count Across Working Day",
        )
        axes[2][0].set(
        xlabel="Hour of the day",
        ylabel="Count",
        title="Box Plot On Count Across Hour of the Day",
        )
        axes[2][1].set(
        ylabel="Count",
        title="Box Plot On Count",
        )
        axes[3][0].set(
        ylabel="Count",
        title="Box Plot On Count vs weekday",
        )            
        axes[3,1].set_axis_off()

        
    def outlier_removal(df):
        """
        This function removes the outliers
        
        Argument:
        df: DataFrame

        """
        try:
            df_wo_outliers = df[
                np.abs(df['count'] - df['count'].mean()) <= (3 * df['count'].std())
            ]
            print('Shape Of the DataFrame Before Outliers: ', df.shape)
            print('Shape Of the DataFrame After Outliers: ', df_wo_outliers.shape)
            print('Number of Outliers: ',df.shape[0] - df_wo_outliers.shape[0])
            return df_wo_outliers
        except Exception as e:
            raise e

    @staticmethod
    def normalisation(df):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(20, 10)
        sns.histplot(x='count', data=df, ax=axes[0])
        df['count'] = np.log(df['count'])
        sns.histplot(x='count', data=df, ax=axes[1])
        axes[0].set(title="Before transformation")
        axes[1].set(title="After transformation")
        return df


    def corr_analysis(corr_mat):
        """
        This function is used for correlation analysis of numerical features.
        
        Argument:
        df: DataFrame

        """
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)
        sns.heatmap(corr_mat, 
                    vmax=0.8, 
                    square=True, 
                    annot=True
                   )


    @staticmethod
    def exp_data_analysis(df):
        """
        This function does exploratory data analysis on dependent variable with 
        respect to independent variable.
        
        Argument:
        df: DataFrame 
        
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.pointplot(data=df, x='month', y='count', ax=ax)
        plt.title('Fig 1: Bike rental count across months', fontsize=20)
        plt.rcParams['axes.labelsize'] = 50

        plt.rcParams['axes.labelsize'] = 20
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.pointplot(data=df, x='hour', y='count', hue='weekday', ax=ax)
        plt.title('Fig 2: Rental bike count across weekdays', fontsize=20)

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.pointplot(data=df, x='hour', y='count', hue='season', ax=ax)
        plt.title('Fig 3: Count of rented bikes across season', fontsize=20)

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.pointplot(data=df, x='hour', y='count', hue='weather', ax=ax)
        plt.title(
            'Fig 4: count of rental bikes during different weathers', fontsize=20
        )

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.pointplot(data=df, x='hour', y='casual', hue='weekday', ax=ax)
        plt.title(
            'Fig 5: count of rental bikes across days of the week for unregistered users',
            fontsize=20
        )

        fig, ax = plt.subplots(figsize=(20, 10))
        sns.pointplot(data=df, x='hour', y='registered', hue='weekday', ax=ax)
        plt.title(
            'Fig 6: count of rental bikes across days of the week for unregistered users',
            fontsize=20
        )
   

    def one_hot_encoding(df):
        """
        This function used to encode the categorial variable.

        Arguments:
        df: DataFrame

        """
        columns = ['season', 'weather', 'holiday', 'weekday', 'workingday','month','year','hour']
        for column in columns:
            df_encoded = pd.get_dummies(df[column], prefix=column, drop_first=True)
            df = pd.concat([df, df_encoded], axis=1)
        df = df.drop(columns, axis=1)
        return df

    def split_into_dep_target(
        df, target_column
        ):
        """
        This function performs the spliting of data into dependent and independent variables
        
        Arguments:
        df: pandas DataFrame that has the updated main data
        
        """
        x = df.drop([target_column], axis=1)

        y = df[target_column]

        return x, y
