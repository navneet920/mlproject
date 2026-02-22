import os
import sys
import pandas as pd


from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict_data(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocess_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocess_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)

            return pred
        except Exception as e:
            raise CustomException(e,sys)
