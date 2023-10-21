from contextlib import asynccontextmanager
import os

import numpy as np

from fastapi import FastAPI

from .utils import load_pickle
from .schemas import MLRequest
from models.ML_classifier_model import CustomMLmodel


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath,'..','assets/model1.pickle')
    ml_models["model1"] = load_pickle(filepath)
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def root():
    ''' Standard health check.'''
    return {"pong": "Hello World"}

@app.post("/invocations")
async def predict(request:MLRequest):
    ''' Function used to make ML inferences.'''
    X_array = np.array([request.covar]).reshape(-1,1)
    return ml_models["model1"].predict(X_array).item(0)

