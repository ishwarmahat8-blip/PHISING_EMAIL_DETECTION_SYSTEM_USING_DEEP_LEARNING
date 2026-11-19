import re
import time
import torch
import numpy as np
import shap
from typing import Dict, Any, List, Optional
from model_loader import model as default_model, tokenizer as default_tokenizer, device, CLASS_LABELS

class SHAPTextExplainer:
    """
    Explainable AI engine using SHAP (SHapley Additive exPlanations) for Transformer-based email classification.
    Uses cooperative game-theoretic Shapley values to compute exact local additive feature attributions
    directly aligned with the model's native tokenizer.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model or default_model
        self.tokenizer = tokenizer or default_tokenizer
        self.device = device
        # Use native tokenizer masker to preserve exact tokenization without altering whitespace/newlines
        self.masker = shap.maskers.Text(self.tokenizer)
        self.class_names = [
            "Legitimate" if "LEGIT" in str(CLASS_LABELS.get(index, "")).upper()
            else "Phishing"
            if "PHISH" in str(CLASS_LABELS.get(index, "")).upper()
            or "FRAUD" in str(CLASS_LABELS.get(index, "")).upper()
            else str(CLASS_LABELS.get(index, f"CLASS_{index}"))
            for index in range(len(CLASS_LABELS))
        ]

    def _predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batched prediction function matching the exact preprocessing pipeline (max_length=256, truncation).
        Returns class probabilities for [Legitimate (0), Phishing (1)].
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # DistilBERT does not take token_type_ids; remove if present
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        return probabilities

    def explain(
        self,
        text: str,
        target_class_idx: int = 1,
        max_evals: int = 200,
        tolerance: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Generates SHAP feature attributions for a given email text and target class.

        Args:
            text: Combined email text (Subject + Body).
            target_class_idx: 0 for Legitimate, 1 for Phishing (typically the predicted class).
            max_evals: Maximum number of coalition evaluations for SHAP (default: 200).
            tolerance: Maximum allowable discrepancy for the additive efficiency test.

        Returns:
            Structured dictionary containing features with exact character offsets, importance scores,
            base value, and execution metrics.
        """
        start_time = time.perf_counter()

        try:
            # Direct model probability on the unmasked text for fidelity verification
            direct_prob = float(self._predict_proba_batch([text])[0, target_class_idx])

            # Build SHAP explainer with the model's native tokenizer masker
            explainer = shap.Explainer(
                self._predict_proba_batch,
                self.masker,
                output_names=self.class_names
            )

            # Compute Shapley values
            shap_values = explainer([text], max_evals=max_evals)

            # Base value is the expected probability when all text tokens are masked
            base_value = float(shap_values.base_values[0, target_class_idx])
            values = shap_values.values[0, :, target_class_idx]

            # Get tokenizer character offset mappings
            enc = self.tokenizer(text, return_offsets_mapping=True, max_length=256, truncation=True)
            offsets = enc["offset_mapping"]
            input_ids = enc["input_ids"]

            raw_token_attributions = []
            structural_attributions = 0.0

            # Step 1: Map raw tokens to character offsets
            for i, (start, end) in enumerate(offsets):
                token_str = self.tokenizer.decode([input_ids[i]])
                val = float(values[i])

                if start == end:
                    # Structural boundary tokens ([CLS], [SEP])
                    structural_attributions += val
                    continue

                raw_token_attributions.append({
                    "token": token_str,
                    "importance": val,
                    "start_char": start,
                    "end_char": end,
                    "slice": text[start:end]
                })

            # Step 2: Aggregate subwords into natural word spans (preserving URLs, punctuation, and newlines)
            aggregated_features: List[Dict[str, Any]] = []
            current_feature = None

            for item in raw_token_attributions:
                start = item["start_char"]
                end = item["end_char"]
                tok = item["token"]
                val = item["importance"]

                if current_feature is not None and (start == current_feature["end_char"] or tok.startswith("##")):
                    # Subword continuation: extend span and sum Shapley attributions
                    current_feature["end_char"] = end
                    current_feature["importance"] += val
                    current_feature["subwords"].append(tok)
                    current_feature["word"] = text[current_feature["start_char"]:end]
                else:
                    if current_feature is not None:
                        aggregated_features.append(current_feature)
                    current_feature = {
                        "word": text[start:end],
                        "importance": val,
                        "start_char": start,
                        "end_char": end,
                        "subwords": [tok]
                    }

            if current_feature is not None:
                aggregated_features.append(current_feature)

            # Step 3: Format features with directional context relative to the target class
            formatted_features: List[Dict[str, Any]] = []
            for feat in aggregated_features:
                importance = feat["importance"]
                if target_class_idx == 1:
                    category = "phishing" if importance > 0 else "legitimate"
                    direction = "supports_phishing" if importance > 0 else "supports_legitimate"
                else:
                    category = "legitimate" if importance > 0 else "phishing"
                    direction = "supports_legitimate" if importance > 0 else "supports_phishing"

                formatted_features.append({
                    "word": feat["word"],
                    "importance": round(importance, 6),
                    "abs_importance": round(abs(importance), 6),
                    "start_char": feat["start_char"],
                    "end_char": feat["end_char"],
                    "category": category,
                    "direction": direction,
                    "subwords": feat["subwords"]
                })

            # Step 4: Verification of Additive Faithfulness (Efficiency Property)
            sum_token_attributions = float(np.sum(values))
            reconstructed_prob = base_value + sum_token_attributions
            discrepancy = abs(reconstructed_prob - direct_prob)
            meets_tolerance = bool(discrepancy <= tolerance)

            top_features = sorted(formatted_features, key=lambda x: x["abs_importance"], reverse=True)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "status": "success",
                "target_class": self.class_names[target_class_idx],
                "target_class_idx": target_class_idx,
                "base_value": round(base_value, 6),
                "base_value_description": (
                    "The model's baseline prediction probability when all input tokens are masked. "
                    "Additive Shapley attributions sum with this base value to reconstruct the target probability."
                ),
                "max_evals": max_evals,
                "execution_time_ms": elapsed_ms,
                "additive_check": {
                    "base_value": round(base_value, 6),
                    "sum_attributions": round(sum_token_attributions, 6),
                    "reconstructed_probability": round(reconstructed_prob, 6),
                    "direct_model_probability": round(direct_prob, 6),
                    "discrepancy": round(discrepancy, 10),
                    "tolerance": tolerance,
                    "meets_tolerance": meets_tolerance
                },
                "structural_attributions": round(structural_attributions, 6),
                "features": formatted_features,          # Sequential order for text highlighting
                "top_features": top_features[:15]        # Top 15 by magnitude
            }

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "error",
                "error_message": f"SHAP explanation generation failed: {str(e)}",
                "target_class": self.class_names[target_class_idx],
                "target_class_idx": target_class_idx,
                "base_value": None,
                "base_value_description": None,
                "max_evals": max_evals,
                "execution_time_ms": elapsed_ms,
                "additive_check": None,
                "features": [],
                "top_features": []
            }

# Singleton instance
shap_explainer = SHAPTextExplainer()
