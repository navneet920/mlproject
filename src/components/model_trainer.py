import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object,evaluate_model

@dataclass()
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Linear_regression":LinearRegression(),
                "Ridge":Ridge(),
                "Decision_tree":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XgbRegressor":XGBRegressor(),
                "catboostRegressor":CatBoostRegressor(),
                "AdaboostRegressor":AdaBoostRegressor(),
                "GradientBoostRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor()
            }

            params={
                "Linear_regression":{},
                "Ridge":{
                    "alpha":[0.001,0.0001,0.01,0.1],
                    "fit_intercept":[True,False]
                },
                "Decision_tree":{
                    "criterion":['squared_error','friedman_mse','absolute_error', 'poisson'],
                    "max_depth":[10,50,500],
                    "min_samples_split":[1,2,4,5]
                },
                "KNeighborsRegressor":{
                    "n_neighbors":[3,5,7,9]
                },
                "XgbRegressor":{
                    "learning_rate":[0.1,1,0.001],
                    "max_depth":[10,50,100,500],
                    "n_estimators":[8,16,32,64]
                },
                "catboostRegressor":{
                    "learning_rate":[1,0.001,0.1],
                    "depth":[6,8,10],
                    "iterations":[30,50,100]
                },
                "AdaboostRegressor":{
                    "learning_rate": [1, 0.001, 0.1],
                    "n_estimators": [8, 16, 32, 64]
                },
                "GradientBoostRegressor":{
                    "learning_rate": [1, 0.001, 0.1],
                    "n_estimators": [8, 16, 32, 64]
                },
                "RandomForestRegressor":{
                    "n_estimators":[10,50,80,100],
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                }
            }
            logging.info("Hyperparameter Set for all models")


            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,
                                    X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_squared=r2_score(predicted,y_test)

            return r2_squared


        except Exception as e:
            raise CustomException(e,sys)

