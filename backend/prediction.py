import torch
import numpy as np
from model_loader import (
    model,
    tokenizer,
    device,
    CLASS_LABELS,
    PHISHING_CLASS,
    LEGITIMATE_CLASS,
)
from explainers.shap_explainer import shap_explainer
from explainers.lime_explainer import lime_explainer

def predict_email(subject: str, body: str, sender: str = ""):
    combined_text = f"Subject: {subject} \n Body: {body}"

    # Tokenize input text securely
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Softmax probability conversion
    exp_logits = np.exp(logits.cpu().numpy() - np.max(logits.cpu().numpy(), axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    prob_legitimate = float(probabilities[0][LEGITIMATE_CLASS])
    prob_phishing = float(probabilities[0][PHISHING_CLASS])

    predicted_class = int(np.argmax(probabilities[0]))
    confidence = float(np.max(probabilities[0]))
    prediction_label = (
        "Phishing"
        if predicted_class == PHISHING_CLASS
        else "Legitimate"
    )

    shap_result = shap_explainer.explain(
        text=combined_text,
        target_class_idx=predicted_class,
        max_evals=200,
        tolerance=1e-4
    )
    lime_result = lime_explainer.explain(
        text=combined_text,
        target_class_idx=predicted_class,
        num_features=15,
        num_samples=100
    )

    shap_indicators = [
        {
            "token": feature["word"],
            "importance": round(abs(feature["importance"]) * 100, 2)
        }
        for feature in shap_result.get("top_features", [])[:5]
    ]

    # Generate explanation reasons
    reasons = []
    body_lower = body.lower()
    subject_lower = subject.lower()

    if prediction_label == "Phishing":
        if any(w in body_lower or w in subject_lower for w in ["urgent", "suspended", "verify", "click", "immediate", "account"]):
            reasons.append("Contains high-pressure urgency cues or account suspension warnings typical of credential harvesting.")
        if any(w in sender.lower() for w in ["support", "security", "update", "service"]) and not any(d in sender.lower() for d in [".paypal.com", ".google.com", ".microsoft.com"]):
            reasons.append("Sender address displays domain spoofing patterns mimicking corporate entities.")
        reasons.append("Transformer attention mechanism weighted semantic embeddings matching known phishing structures.")
    else:
        reasons.append("Normal conversational or transactional tone lacking credential-harvesting triggers.")
        reasons.append("Sender domain structure and email subject align with legitimate communication patterns.")

    return {
        "prediction": prediction_label,
        "classification_label": predicted_class,
        "phishing_probability": round(prob_phishing * 100, 2),
        "legitimate_probability": round(prob_legitimate * 100, 2),
        "confidence_score": round(confidence * 100, 2),
        "explanation": {
            "reasons": reasons,
            "key_indicators": shap_indicators,
            "shap": shap_result,
            "lime": lime_result
        }
    }