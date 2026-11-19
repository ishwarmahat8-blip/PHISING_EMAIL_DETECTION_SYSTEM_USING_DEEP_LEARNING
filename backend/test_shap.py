import sys
from pathlib import Path

# Add backend directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from explainers.shap_explainer import shap_explainer
from model_loader import model, tokenizer, device
import torch

def run_shap_evaluation_budget_test(title: str, subject: str, body: str, expected_label: str):
    print("=" * 80)
    print(f"TEST CASE: {title}")
    print(f"Subject:  {subject}")
    print(f"Body:     {body}")
    print(f"Expected: {expected_label}")
    print("-" * 80)

    combined_text = f"Subject: {subject} \n Body: {body}"

    # 1. Model inference
    inputs = tokenizer(
        combined_text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    prob_legitimate = float(probs[0])
    prob_phishing = float(probs[1])
    predicted_class = 1 if prob_phishing >= 0.5 else 0
    predicted_label = "Phishing" if predicted_class == 1 else "Legitimate"

    print(f"Model Prediction: {predicted_label} (Class Index: {predicted_class})")
    print(f"  P(Legitimate): {prob_legitimate * 100:.4f}%")
    print(f"  P(Phishing):   {prob_phishing * 100:.4f}%")

    # 2. Evaluate across budgets: 200 and 500
    for max_evals in [200, 500]:
        print(f"\n--- Testing SHAP Budget: max_evals = {max_evals} ---")
        shap_result = shap_explainer.explain(
            text=combined_text,
            target_class_idx=predicted_class,
            max_evals=max_evals,
            tolerance=1e-5
        )

        assert shap_result["status"] == "success", f"SHAP failed: {shap_result.get('error_message')}"

        check = shap_result["additive_check"]
        print(f"Execution Time:      {shap_result['execution_time_ms']:.2f} ms")
        print(f"Base Value:          {check['base_value']:.6f}")
        print(f"Sum of Attributions: {check['sum_attributions']:+.6f}")
        print(f"Reconstructed Prob:  {check['reconstructed_probability']:.6f}")
        print(f"Direct Model Prob:   {check['direct_model_probability']:.6f}")
        print(f"Discrepancy:         {check['discrepancy']:.10f}")
        print(f"Tolerance (1e-5):    {check['tolerance']}")
        print(f"Meets Tolerance:     {check['meets_tolerance']}")

        # Enforce strict tolerance assertion
        assert check["meets_tolerance"], (
            f"FAIL: Discrepancy {check['discrepancy']} exceeds tolerance {check['tolerance']}!"
        )

        # 3. Verify character alignment on top features
        print("\nTop 5 Influential Features & Exact Text Alignment:")
        for feat in shap_result["top_features"][:5]:
            start = feat["start_char"]
            end = feat["end_char"]
            extracted = combined_text[start:end]
            match = (extracted == feat["word"])
            print(f"  - Word: {repr(feat['word']):20} | Importance: {feat['importance']:+.6f} | "
                  f"Direction: {feat['direction']:20} | Span: ({start:3d}, {end:3d}) | Exact Match: {match}")
            assert match, f"Character offset mismatch for '{feat['word']}'! Extracted: '{extracted}'"

    print(f"\n>>> ALL CHECKS PASSED FOR: {title} <<<")

if __name__ == "__main__":
    # Test 1: Phishing email with urgency and account suspension
    run_shap_evaluation_budget_test(
        title="High-Pressure Phishing Email",
        subject="URGENT: Your account has been suspended! Verify now.",
        body="Dear customer, unauthorized login detected. Click here immediately to restore your account access or it will be permanently closed.",
        expected_label="Phishing"
    )

    # Test 2: Legitimate business email with meeting schedule
    run_shap_evaluation_budget_test(
        title="Legitimate Business Email",
        subject="Project meeting scheduled for tomorrow at 10 AM",
        body="Hi team, please find attached the agenda for our weekly sprint review. Let me know if you have any questions.",
        expected_label="Legitimate"
    )

    # Test 3: Phishing email with suspicious URL and special characters
    run_shap_evaluation_budget_test(
        title="Phishing Email with Suspicious URL",
        subject="Action Required: Security Alert on Your Bank Account",
        body="We detected suspicious activity. Please visit https://secure-bank-update.com/login?ref=user to confirm your identity.",
        expected_label="Phishing"
    )
