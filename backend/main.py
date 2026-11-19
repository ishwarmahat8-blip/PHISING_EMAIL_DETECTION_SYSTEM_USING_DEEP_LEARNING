from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import EmailInput, PredictionResponse
from prediction import predict_email
from model_loader import CLASS_LABELS

app = FastAPI(
    title="AI Email Phishing Detection API",
    description="MSc Research Project API powered by fine-tuned DistilBERT",
    version="1.0.0"
)

# Configure CORS so your Next.js frontend can communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Status"])
def health_check():
    return {
        "status": "online",
        "model": "DistilBERT Phishing Classifier",
        "model_directory": "model",
        "labels": CLASS_LABELS,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def classify_email(email: EmailInput):
    try:
        # Pass subject, body, and sender to match the updated prediction engine
        result = predict_email(subject=email.subject, body=email.body, sender=email.sender)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))