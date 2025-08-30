import logging
import pickle
import os
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models: dict, param_grids):
    """
    Trains and evaluates multiple models.
    Stores both train and test R² scores in a dictionary.
    """
    try:
        report = {}
        for name, model in models.items():
            if name in param_grids:
                model = GridSearchCV(
                    estimator=model, param_grid=param_grids[name], scoring="r2", cv=5
                )
            model.fit(X_train, y_train)

            # If GridSearchCV → keep best estimator
            if isinstance(model, GridSearchCV):
                logging.info(f"{name} -> Best Params: {model.best_params_}")
                model = model.best_estimator_

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store scores in dictionary
            report[name] = {
                "train_r2": train_model_score,
                "test_r2": test_model_score,
                "model": model,
            }

            # Log results
            logging.info(
                f"Hyper Tune-> Model: {name} | Train R²: {train_model_score:.4f} | Test R²: {test_model_score:.4f}"
            )

        return report

    except Exception as e:
        raise CustomException(e, sys)
