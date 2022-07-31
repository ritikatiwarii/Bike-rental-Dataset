try:
    from typing import Tuple
    import numpy as np
    import pandas as pd
    from sklearn import ensemble
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.model_selection import RandomizedSearchCV
    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor
except ImportError as error:
    raise error


class Model:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        train_test_ratio: float,
        pretrained: bool = False,
        model_path: str = None,
    ) -> None:
        """
        This function initializes the target,independent variables, train and test data, and the model.
        
        Arguments:
        X: Independent variables.
        y: Target variable.
        train_test_ratio: the test train split ratio.
        
        """

        self.X = X
        self.y = y
        self.train_test_ratio = train_test_ratio
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_test_split()
        if pretrained:
            self.model = XGBRegressor()
            self.model.load_model(model_path)
        else:
            self.model = XGBRegressor()

    def train_test_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        This function split the data into train and test with a ratio.
        
        Arguments:
        X_train,y_train: splitting the training data set
        X_test,y_test: splitting testing data set
        
        Returns:
        X_train, y_train, X_test, y_test: Returns the splitted data set

        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=self.train_test_ratio, random_state=42
        )
        return X_train, y_train, X_test, y_test

    def fit(self) -> None:
        """
        This function used to fit the model for X_train and y_train(training dataset)

        """

        self.model.fit(self.X_train, self.y_train)

    def save_model(self, path):
        """
        This function is used to save the model
        """
        self.model.save_model(path)

    def predict(self) -> pd.DataFrame:
        """
        This function used for prediction purpose with X_test
        
        Returns:
        pred - returns the prdicted value
        
        """

        pred = self.model.predict(self.X_test)

        return pred

    def predict_one(self, test):
        """
        This function used predict for one sample point.
        Returns:
        pred - returns the prdicted value
        """
        pred = self.model.predict(test)[0]
        return pred

    def hyper_parameter_tuning(self, hyperparameter_grid) -> dict:
        """
        hyper_parameter_tuning used for estimating the best parameters for the model.
        
        Argument:
        hyperparameter_grid: assigning the hyparameter values to identify the optimimum one.
        
        Returns:
        best_estimator_: return the best estimator, estimated from the RandomizedsearchCV.

        """

        random_cv = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=hyperparameter_grid,
            cv=5,
            n_iter=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=10,
            return_train_score=True,
            random_state=42,
        )
        random_cv.fit(self.X_train, self.y_train)

        print(" Best Estimator for XGBoost. ")
        print(random_cv.best_estimator_)

        return random_cv.best_estimator_

    def show_results(self) -> None:
        """
        Show results 
        This function is used to plot predictions for actual predictions,
        versus model predictions
        """
        results = pd.DataFrame(
            {"y_test": self.y_test, "model prediction": self.model.predict(self.X_test)}
        )
        return results

    def mae(self, y_pred: pd.DataFrame) -> Tuple[np.array, np.array]:
        """
        This function performs the evaluation of the model in terms of mae metrice
        
        Arguments :
        y_pred: predicted y_values.
        
        Returns:
        mae( y_test, y_pred): result of the calculated mae
        
        """

        y_test, y_pred = np.array(self.y_test), np.array(y_pred)

        return mae(y_test, y_pred)
