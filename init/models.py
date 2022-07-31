import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import RandomizedSearchCV
from sklearn import ensemble
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class models:
    def __init__(self):
        """
        Initialisation
        
        """

    def train_test_split(x,
                         y,
                         ratio
                        ):
        """
        This function split the data into train and test with a ratio.
        
        Arguments:
        independent variable,
        dependent variable,
        ratio

        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=ratio, random_state=42
        )
        return x_train, x_test, y_train, y_test
    
    
    def model_lr(x_train,y_train,x_test,y_test):
        """
        Function to implement linear regression
        
        Arguments: x_train,
                   y_train,
                   x_test

        """
        model_reg = LinearRegression()
        
        # Running the model on Training Data
        lin_reg=model_reg.fit(x_train,y_train)
        # Predicting on the test data set
        y_pred_reg=lin_reg.predict(x_test)
        # Finding the mean absolute  error
        print('Mean Absolute error for linear regression: ',mae(y_test, y_pred_reg))
        
    def model_rf(x_train,y_train,x_test,y_test):
        """
        Function to implement random forest algorithm

        Arguments: x_train,
               y_train,
               x_test

        """
        model_rf = RandomForestRegressor(random_state=42)

        # Selecting best max_depth, maximum features, split criterion and number of trees
        param_dist = {'max_depth': [2,4,6,8,10],
                      'bootstrap': [True, False],
                      'max_features': ['auto', 'sqrt', 'log2',None],
                      "n_estimators" : [100 ,200 ,300 ,400 ,500]
                     }
        cv_randomForest = RandomizedSearchCV(model_rf, cv = 10,
                             param_distributions = param_dist, 
                             n_iter = 10)

        cv_randomForest.fit(x_train, y_train)
        print('Best Parameters using random search: \n', 
              cv_randomForest.best_params_)
        
        model_rf.set_params( max_features = None,
                       max_depth =10 ,
                       n_estimators = 100,
                       bootstrap = True
                      )
        #Fit the random forest model to train dataset
        model_rf.fit(x_train, y_train)

        # Predict on the test dataset
        y_pred_rf = model_rf.predict(x_test)
        # Calculate the absolute errors
        print('Mean Absolute error for random forest model: ',mae(y_test, y_pred_rf))

        
    def model_xgb(x_train,y_train,x_test,y_test,hyperparameter_grid):
        """
        Function to implement XGBOOST

        Arguments: x_train,
               y_train,
               x_test,
               hyperparameter_grid

        """
        #hyperparameter tuning for estimating the best parameters for the model
        xgb_model = XGBRegressor(silent=True)

        random_cv = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=hyperparameter_grid,
            cv=5,
            n_iter=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
            random_state=42,
        )
        random_cv.fit(x_train,y_train)
        print('Best Estimator for XGBoost: \n',random_cv.best_estimator_)

        best_estimator = random_cv.best_estimator_

        #Fitting the model on the training dataset    
        best_estimator.fit(x_train, y_train)

        #Making predictions with the model on test data
        y_pred_xg = best_estimator.predict(x_test)

        #printing the mean absolute percentage error
        print('Mean Absolute error for xgboost: ',mae(y_test, y_pred_xg))
        

