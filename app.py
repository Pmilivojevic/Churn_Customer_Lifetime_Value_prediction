from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
# import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from churn_pred.pipelines.stage_06_prediction_pipeline import PredictionPipeline
from churn_pred import logger
import io

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Validate file extension
        if not file.filename.endswith('.csv'):
            logger.error("Invalid file type: Only CSV files are supported")
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content
        contents = await file.read()
        csv_file = io.StringIO(contents.decode('utf-8'))
        
        # Validate file is not empty
        if not contents:
            logger.error("Uploaded CSV file is empty")
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        obj = PredictionPipeline()
        lg_pred_proba, xgb_pred_proba = obj.predict(csv_file)
        
        # Format response
        response = {
            "logistic_regression_probabilities": lg_pred_proba.tolist(),
            "xgboost_probabilities": xgb_pred_proba.tolist()
        }
        logger.info("Prediction successful")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
