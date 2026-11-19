import time
from typing import Any, Dict, List

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from model_loader import model as default_model, tokenizer as default_tokenizer, device, CLASS_LABELS


class LIMETextExplainer:
    def __init__(self, model=None, tokenizer=None):
        self.model = model or default_model
        self.tokenizer = tokenizer or default_tokenizer
        self.device = device
        self.class_names = [
            "Legitimate" if "LEGIT" in str(CLASS_LABELS.get(index, "")).upper()
            else "Phishing"
            if "PHISH" in str(CLASS_LABELS.get(index, "")).upper()
            or "FRAUD" in str(CLASS_LABELS.get(index, "")).upper()
            else str(CLASS_LABELS.get(index, f"CLASS_{index}"))
            for index in range(len(CLASS_LABELS))
        ]

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        return probabilities

    def explain(
        self,
        text: str,
        target_class_idx: int = 1,
        num_features: int = 15,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        try:
            explainer = LimeTextExplainer(
                class_names=self.class_names,
                bow=False,
                random_state=42
            )
            explanation = explainer.explain_instance(
                text,
                self._predict_proba,
                labels=(target_class_idx,),
                num_features=num_features,
                num_samples=num_samples
            )

            features = []
            for feature, weight in explanation.as_list(label=target_class_idx):
                weight = float(weight)
                features.append({
                    "word": feature,
                    "weight": round(weight, 6),
                    "abs_weight": round(abs(weight), 6),
                    "direction": (
                        "supports_phishing" if target_class_idx == 1 and weight > 0
                        else "supports_legitimate" if target_class_idx == 0 and weight > 0
                        else "supports_opposite_class"
                    )
                })

            probabilities = self._predict_proba([text])[0]
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            local_pred = explanation.local_pred[0] if hasattr(explanation, "local_pred") else None

            return {
                "status": "success",
                "target_class": self.class_names[target_class_idx],
                "target_class_idx": target_class_idx,
                "num_features": num_features,
                "num_samples": num_samples,
                "execution_time_ms": elapsed_ms,
                "local_prediction": round(float(local_pred), 6) if local_pred is not None else None,
                "direct_model_probability": round(float(probabilities[target_class_idx]), 6),
                "features": features,
                "top_features": sorted(features, key=lambda item: item["abs_weight"], reverse=True)
            }
        except Exception as error:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "error",
                "error_message": f"LIME explanation generation failed: {str(error)}",
                "target_class": self.class_names[target_class_idx],
                "target_class_idx": target_class_idx,
                "num_features": num_features,
                "num_samples": num_samples,
                "execution_time_ms": elapsed_ms,
                "features": [],
                "top_features": []
            }


lime_explainer = LIMETextExplainer()
