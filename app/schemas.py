from pydantic import BaseModel

class MLRequest(BaseModel):
    covar: float