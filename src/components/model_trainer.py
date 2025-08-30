import os
from re import X
import sys
from dataclasses import dataclass
import dill

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trainer_model_filepath = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Spliting Training And Test Input Data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Log shapes
            logging.info(
                f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}"
            )
            logging.info(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

            # Dictionary of models without parameters
            models = {
                "LinearRegression": LinearRegression(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
            }

            param_grids = {
                "RandomForest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 5],
                    "min_samples_split": [2, 5],
                },
                "ElasticNet": {
                    "alpha": [0.01, 0.1, 1.0, 10.0],  # Regularization strength
                    "l1_ratio": [
                        0.1,
                        0.5,
                        0.9,
                    ],  # Mix between L1 (Lasso) and L2 (Ridge)
                },
            }

            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param_grids=param_grids,
            )

            # Find best model based on test R²
            best_model_name = max(
                model_report, key=lambda x: model_report[x]["test_r2"]
            )
            best_model_score = model_report[best_model_name]["test_r2"]

            # best_model = models[best_model_name]
            # ✅ Get the trained model from report, not from models

            best_model = model_report[best_model_name]["model"]

            if best_model_score < 0.6:
                logging.info("No Best Model Found!")
                raise CustomException("No Best Model Found!")

            logging.info(f"Best Model: {best_model_name} | Score: {best_model_score}")

            save_object(
                filepath=self.model_trainer_config.trainer_model_filepath,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, predicted)

            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)
