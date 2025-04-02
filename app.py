from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import pandas as pd


app = FastAPI()

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_order = ['Gender', 'Age', 'Height', 'Heart_Rate', 'Body_Temp']


class ScoringItem(BaseModel):
    Gender: int  
    Age: float   
    Height: float  
    Heart_Rate: float 
    Body_Temp: float  

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    scaled_df = scaler.transform(df)
    yhat = model.predict(scaled_df)[0]
    return {"prediction": float(yhat)}


