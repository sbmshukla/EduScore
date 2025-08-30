import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessor with numeric median scaling and
        categorical most-frequent imputation + OHE + scaling.
        """
        try:
            # Define columns
            numeric_column = ["reading_score", "writing_score"]
            categorical_column = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numeric pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder()),
                    (
                        "scaler",
                        StandardScaler(with_mean=False),
                    ),  # Important for sparse matrix
                ]
            )

            logging.info("Numeric Column Scaling Completed")
            logging.info("Categorical Columns Encoding Completed")

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", num_pipeline, numeric_column),
                    ("categorical_pipeline", cat_pipeline, categorical_column),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data Completed")

            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numeric_column = ["reading_score", "writing_score"]
            categorical_column = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            input_feature__train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature__train_df = train_df[target_column]

            input_feature__test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature__test_df = test_df[target_column]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature__train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature__test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature__train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature__test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            # logging.info(f"Train Shape: {train_arr.shape}")
            # logging.info(f"Test Shape: {test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
