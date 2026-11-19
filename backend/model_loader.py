import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from typing import Tuple, Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model registry cache to prevent redundant re-loading
_LOADED_MODELS: Dict[str, Dict[str, Any]] = {}


def load_model_metadata(model_dir: Path) -> Dict[str, Any]:
    for filename in ("training_metadata.json", "label_mapping.json", "config.json"):
        metadata_path = model_dir / filename
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
    return {}


def resolve_class_labels(model: Any, model_dir: Path) -> Dict[int, str]:
    config_labels = getattr(model.config, "id2label", None)
    if config_labels:
        return {int(index): str(label) for index, label in config_labels.items()}

    metadata = load_model_metadata(model_dir)
    label_mapping = metadata.get("labels", metadata)
    if isinstance(label_mapping, dict):
        return {int(index): str(label) for index, label in label_mapping.items()}
    return {0: "LEGITIMATE", 1: "FRAUDULENT"}


def get_phishing_class(labels: Dict[int, str]) -> int:
    for index, label in labels.items():
        normalized = label.upper()
        if "PHISH" in normalized or "FRAUD" in normalized:
            return index
    return 1


def get_legitimate_class(labels: Dict[int, str]) -> int:
    for index, label in labels.items():
        normalized = label.upper()
        if "LEGIT" in normalized:
            return index
    return 0

def load_model_and_tokenizer(model_name: str = "distilbert", model_path: Path = None) -> Tuple[Any, Any]:
    """
    Loads and caches a sequence classification model and its corresponding tokenizer.
    Designed to support DistilBERT and easily extend to DeBERTa or other architectures.
    """
    key = model_name.lower()
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]["model"], _LOADED_MODELS[key]["tokenizer"]

    target_dir = Path(model_path) if model_path is not None else DEFAULT_MODEL_DIR
    if not target_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {target_dir}. Expected config.json and model.safetensors."
        )

    print(f"Loading {model_name} model and tokenizer from: {target_dir}")
    try:
        # AutoTokenizer loads the exact tokenizer matching tokenizer_config.json
        tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(target_dir), local_files_only=True)
        model.to(device)
        model.eval()
        class_labels = resolve_class_labels(model, target_dir)
        
        _LOADED_MODELS[key] = {
            "model": model,
            "tokenizer": tokenizer,
            "class_labels": class_labels,
            "phishing_class": get_phishing_class(class_labels),
            "legitimate_class": get_legitimate_class(class_labels),
            "model_dir": target_dir
        }
        print(f"Model '{model_name}' successfully loaded on device: {device}")
        return model, tokenizer
    except Exception as e:
        print(f"ERROR: Failed to load {model_name} model from {target_dir}. Ensure config.json and model weights exist.")
        raise e

# Initialize the default DistilBERT model for immediate use and backward compatibility
model, tokenizer = load_model_and_tokenizer("distilbert", DEFAULT_MODEL_DIR)
CLASS_LABELS = resolve_class_labels(model, DEFAULT_MODEL_DIR)
PHISHING_CLASS = get_phishing_class(CLASS_LABELS)
LEGITIMATE_CLASS = get_legitimate_class(CLASS_LABELS)
