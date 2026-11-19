from pydantic import BaseModel, Field
from typing import Dict, Any

class EmailInput(BaseModel):
    sender: str = Field(..., alias="from", description="Sender email address")
    recipient: str = Field(..., alias="to", description="Recipient email address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    prediction: str
    classification_label: int
    phishing_probability: float
    legitimate_probability: float
    confidence_score: float
    explanation: Dict[str, Any]