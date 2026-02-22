from fastapi import FastAPI
import sys
from src.exception import CustomException
from src.schemas.student_schema import InputData

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.pipeline.predict_pipeline import PredictPipeline
app=FastAPI()


@app.get("/")
def home():
    return {"message":"welcome to student performance predictor"}

@app.post("/predict")
def predict(data:InputData):
    try:

        # convert pydantic → dict
        data_dict = dict(data)

        # convert dict → dataframe
        df = pd.DataFrame([data_dict])

        # call pipeline
        pipeline = PredictPipeline()

        result = pipeline.predict_data(df)

        return {

            "prediction": int(result[0])

        }

    except Exception as e:
        raise CustomException(e,sys)