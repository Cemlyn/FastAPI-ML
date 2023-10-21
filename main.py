from app.api import app
import uvicorn
from models.ML_classifier_model import CustomMLmodel

if __name__ == "__main__":
    uvicorn.run(app, port=8080, log_level="info")
