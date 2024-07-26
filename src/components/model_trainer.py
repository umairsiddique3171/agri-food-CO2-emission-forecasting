import os 
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models,save_report


@dataclass
class ModelTrainerConfig:
    models_training_report_path = os.path.join("artifacts","training_report.csv")
    trained_model_path = os.path.join("artifacts","model.p")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            logging.info("Feature_Target_Split for Train and Test Data completed")
            models = {
                "AdaBoost Regressor":AdaBoostRegressor(),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "XGBoost Regressor":XGBRegressor()
            }
            report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            save_report(report=report,path=self.model_trainer_config.models_training_report_path)
            logging.info("Models Training Report saved in artifacts")
            best_model_score = max(sorted([v[1] for v in list(report.values())]))
            best_model_name = list(report.keys())[[v[1] for v in list(report.values())].index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"{best_model_name} : {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            logging.info(f"{best_model_name} saved in artifacts")
            return best_model_score
        
        except Exception as e:
            raise CustomException(e,sys)