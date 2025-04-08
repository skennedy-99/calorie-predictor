from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn


app = FastAPI()

model = joblib.load('catboost_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_order = ['Heart_Rate', 'Body_Temp', 'Duration', 'BMI', 'Gender', 'Age']


class ScoringItem(BaseModel):
    Heart_Rate: float
    Body_Temp: float
    Duration: float
    BMI: float
    Gender: int  
    Age: float   
 

@app.post('/predict')
async def scoring_endpoint(item: ScoringItem):
    input_data = item.dict()
    df = pd.DataFrame([input_data])
    df = df[feature_order]
    scaled_df = scaler.transform(df)
    yhat = model.predict(scaled_df)[0]
    return {"prediction": float(yhat)}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)