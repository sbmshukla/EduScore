import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


from src.logger import logging


class PredictPipeline:
    def predict(self, features):
        try:
            logging.info("Starting prediction pipeline...")

            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            logging.info(f"Loading preprocessor from: {preprocessor_path}")
            preprocessor = load_object(filepath=preprocessor_path)

            logging.info(f"Loading model from: {model_path}")
            model = load_object(filepath=model_path)

            logging.info("Transforming input features...")
            data_scaled = preprocessor.transform(features)

            logging.info("Generating predictions...")
            preds = model.predict(data_scaled)

            logging.info(f"Prediction completed. Predictions: {preds}")
            return preds

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
