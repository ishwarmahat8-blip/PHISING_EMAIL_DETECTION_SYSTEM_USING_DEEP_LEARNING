import sys
import time
import json
import numpy as np
from pathlib import Path

# Add backend directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from explainers.shap_explainer import shap_explainer, SHAPTextExplainer
from model_loader import model, tokenizer, device
import torch

def predict_email_raw(subject: str, body: str):
    combined_text = f"Subject: {subject} \n Body: {body}"
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

    return {
        "combined_text": combined_text,
        "prob_legitimate": prob_legitimate,
        "prob_phishing": prob_phishing,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label
    }

def run_test_suite():
    print("=" * 90)
    print("PHASE 5: INDEPENDENT SHAP COMPREHENSIVE TEST SUITE")
    print("=" * 90)

    test_cases = [
        {
            "id": "TC1",
            "name": "Short Email",
            "subject": "Urgent notice",
            "body": "Click here",
            "expected_human_label": "Phishing"
        },
        {
            "id": "TC2",
            "name": "Empty or Nearly Empty Body",
            "subject": "Security verification code",
            "body": "",
            "expected_human_label": "Ambiguous/Phishing"
        },
        {
            "id": "TC3",
            "name": "Long Detailed Corporate Email",
            "subject": "Annual Enterprise IT Infrastructure Policy and Compliance Guidelines for Fiscal Year 2026-2027",
            "body": (
                "To all department heads and staff members: Please review the updated corporate data protection "
                "protocols finalized during our executive steering committee meeting yesterday. All workstations "
                "must adhere strictly to multi-factor authentication mandates, and database access logs will be audited "
                "bi-weekly by the internal governance team. If you observe any configuration discrepancies or experience "
                "credential latency, submit an internal ticket via the standard employee intranet portal before Friday afternoon. "
                "Thank you for your continued dedication to organizational operational integrity and data privacy standards."
            ),
            "expected_human_label": "Legitimate"
        },
        {
            "id": "TC4",
            "name": "Multiple Complex URLs & Query Strings",
            "subject": "Action Required: Update Your Billing Authentication",
            "body": (
                "Your subscription expired. Renew immediately at https://auth-service.com/login?token=abc123xyz&ref=alert "
                "or access backup gateway http://192.168.1.100:8080/portal. "
                "For inquiries, visit https://help.internal-domain.org/faq#contact."
            ),
            "expected_human_label": "Phishing"
        },
        {
            "id": "TC5",
            "name": "Punctuation, Numbers, Symbols & Repeated Words",
            "subject": "$$$ SPECIAL OFFER $$$ 50% OFF, OFF, OFF !!!",
            "body": "Call +1-800-555-0199 now! Now, now, now! Save $500.00 today #1 deal: 100% guaranteed. Email: promo@deals-2026.com.",
            "expected_human_label": "Spam/Phishing"
        },
        {
            "id": "TC6",
            "name": "Clearly Legitimate Business Email",
            "subject": "Quarterly Budget Review and Department Objectives",
            "body": "Dear team, please find attached the revised financial projections and agenda for Wednesday's conference. Let me know your feedback by tomorrow evening.",
            "expected_human_label": "Legitimate"
        },
        {
            "id": "TC7",
            "name": "High-Pressure Credential Harvesting Phishing Email",
            "subject": "URGENT: Your account has been suspended! Verify now.",
            "body": "Dear customer, unauthorized login detected. Click here immediately to restore your account access or it will be permanently closed.",
            "expected_human_label": "Phishing"
        },
        {
            "id": "TC8",
            "name": "Potentially Unexpected Model Prediction Case",
            "subject": "Password reset successfully completed",
            "body": "Your account password was updated today. If you made this change, no further action is necessary.",
            "expected_human_label": "Legitimate (Contains Phishing Triggers)"
        }
    ]

    suite_results = []

    for tc in test_cases:
        print("\n" + "-" * 90)
        print(f"RUNNING {tc['id']}: {tc['name']}")
        print(f"Subject:  {tc['subject']}")
        print(f"Body:     {tc['body'][:80]}..." if len(tc['body']) > 80 else f"Body:     {tc['body']}")
        print(f"Expected: {tc['expected_human_label']}")

        # 1. Model inference
        pred = predict_email_raw(tc["subject"], tc["body"])
        text = pred["combined_text"]
        pred_class = pred["predicted_class"]
        pred_label = pred["predicted_label"]

        print(f"Actual Model Output: {pred_label} (Class {pred_class})")
        print(f"  P(Legitimate): {pred["prob_legitimate"] * 100:.2f}% | P(Phishing): {pred["prob_phishing"] * 100:.2f}%")

        # 2. Run SHAP with default 200 evaluations
        t0 = time.perf_counter()
        shap_res = shap_explainer.explain(
            text=text,
            target_class_idx=pred_class,
            max_evals=200,
            tolerance=1e-5
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        status = shap_res["status"]
        print(f"SHAP Status: {status} | Latency: {elapsed_ms} ms")

        if status != "success":
            print(f"ERROR: {shap_res.get('error_message')}")
            suite_results.append({
                "id": tc["id"],
                "name": tc["name"],
                "prediction": pred_label,
                "status": "FAIL",
                "error": shap_res.get("error_message")
            })
            continue

        check = shap_res["additive_check"]
        print(f"Base Value:          {check['base_value']:.6f}")
        print(f"Sum of Attributions: {check['sum_attributions']:+.6f}")
        print(f"Reconstructed Prob:  {check['reconstructed_probability']:.6f}")
        print(f"Direct Model Prob:   {check['direct_model_probability']:.6f}")
        print(f"Discrepancy:         {check['discrepancy']:.10f}")
        print(f"Meets Tol (1e-5):    {check['meets_tolerance']}")

        # 3. Verify character offsets against raw text
        offset_mismatches = 0
        for feat in shap_res["features"]:
            start = feat["start_char"]
            end = feat["end_char"]
            extracted = text[start:end]
            if extracted != feat["word"]:
                offset_mismatches += 1
                print(f"  MISMATCH: feat['word']='{feat['word']}' != text[{start}:{end}]='{extracted}'")

        print(f"Character Offset Alignment: {len(shap_res['features'])} features verified, {offset_mismatches} mismatches.")

        # Top 3 features display
        print("Top 3 Influential Features:")
        for f in shap_res["top_features"][:3]:
            print(f"  - '{f['word']}': {f['importance']:+.5f} ({f['direction']})")

        suite_results.append({
            "id": tc["id"],
            "name": tc["name"],
            "prediction": pred_label,
            "target_class": pred_class,
            "latency_ms": elapsed_ms,
            "discrepancy": check["discrepancy"],
            "meets_tolerance": check["meets_tolerance"],
            "offset_mismatches": offset_mismatches,
            "top_features": [f["word"] for f in shap_res["top_features"][:3]]
        })

    # =========================================================================
    # TEST 6: REPEATED EXECUTIONS FOR EXPLANATION CONSISTENCY (STABILITY)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 6: REPEATED EXECUTION STABILITY TEST (3 Consecutive Runs)")
    print("=" * 90)
    stability_text = (
        "Subject: URGENT: Your account has been suspended! Verify now. \n "
        "Body: Dear customer, unauthorized login detected. Click here immediately to restore your account access or it will be permanently closed."
    )

    run_attributions = []
    run_times = []
    for run_i in range(1, 4):
        t_start = time.perf_counter()
        res = shap_explainer.explain(stability_text, target_class_idx=1, max_evals=200)
        t_run = round((time.perf_counter() - t_start) * 1000, 2)
        run_times.append(t_run)
        # Store dict of word -> importance
        attributions = {f["word"]: f["importance"] for f in res["features"]}
        run_attributions.append(attributions)
        print(f"Run {run_i}: Latency = {t_run} ms, Top 3 = {[f['word'] for f in res['top_features'][:3]]}")

    # Calculate max variance/difference across the 3 runs for every word
    all_words = list(run_attributions[0].keys())
    max_diff_across_runs = 0.0
    for w in all_words:
        scores = [run_attributions[r].get(w, 0.0) for r in range(3)]
        diff = max(scores) - min(scores)
        if diff > max_diff_across_runs:
            max_diff_across_runs = diff

    print(f"Maximum attribution difference across 3 independent runs: {max_diff_across_runs:.10f}")
    is_deterministic = (max_diff_across_runs == 0.0)
    print(f"Is PartitionExplainer deterministic across repeated executions: {is_deterministic}")

    # =========================================================================
    # TEST 7: INVESTIGATION OF EVALUATION BUDGETS (50, 100, 150, 200, 500, 1000)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 7: EVALUATION BUDGET SCALING INVESTIGATION (PartitionExplainer Tree Dynamics)")
    print("=" * 90)

    # Use long email with many tokens to test hierarchical tree saturation
    long_email_text = (
        "Subject: Enterprise Security Alert \n "
        "Body: To all staff members: Please review the updated corporate data protection protocols finalized yesterday. "
        "All workstations must adhere strictly to multi-factor authentication mandates, and database access logs will be audited "
        "bi-weekly by the internal governance team. If you observe any configuration discrepancies, submit a ticket immediately."
    )

    enc_long = tokenizer(long_email_text, truncation=True, max_length=256)
    token_count = len(enc_long["input_ids"])
    print(f"Input Token Count: {token_count} tokens")
    print(f"Theoretical Partition Tree Nodes: 2 * N - 1 = {2 * token_count - 1} nodes")

    budget_results = []
    for budget in [30, 60, 100, 150, 200, 500]:
        t0 = time.perf_counter()
        res_budget = shap_explainer.explain(long_email_text, target_class_idx=1, max_evals=budget)
        lat = round((time.perf_counter() - t0) * 1000, 2)
        top_words = [f["word"] for f in res_budget["top_features"][:3]]
        top_scores = [f["importance"] for f in res_budget["top_features"][:3]]
        disc = res_budget["additive_check"]["discrepancy"]
        budget_results.append({
            "budget": budget,
            "latency_ms": lat,
            "discrepancy": disc,
            "top_words": top_words,
            "top_scores": top_scores
        })
        print(f"Budget = {budget:4d} | Latency = {lat:7.1f} ms | Discrepancy = {disc:.10f} | Top 3: {list(zip(top_words, top_scores))}")

    # =========================================================================
    # TEST 8: ERROR HANDLING (NO FABRICATED EXPLANATIONS)
    # =========================================================================
    print("\n" + "=" * 90)
    print("TEST 8: ERROR RESILIENCE (VERIFY NO DUMMY/FABRICATED RESULTS)")
    print("=" * 90)

    # Create a deliberately broken explainer with a dummy failing model
    class FailingModel:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Simulated GPU Out Of Memory / Engine Failure")

    broken_explainer = SHAPTextExplainer(model=FailingModel())
    err_res = broken_explainer.explain("Subject: Test \n Body: Failure test", target_class_idx=1)

    print(f"Error handling status: '{err_res['status']}'")
    print(f"Error message: '{err_res.get('error_message')}'")
    print(f"Features returned: {err_res['features']} (Length: {len(err_res['features'])})")
    assert err_res["status"] == "error", "Explainer did not report error!"
    assert len(err_res["features"]) == 0, "Explainer returned fabricated features on error!"
    print("Verification Passed: Zero fabricated results returned during failure.")

    # =========================================================================
    # SUMMARY REPORT TABLE
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHASE 5 TEST SUMMARY REPORT")
    print("=" * 90)
    print(f"{'ID':<6} | {'Test Case Name':<35} | {'Pred':<12} | {'Latency':<10} | {'Discrepancy':<12} | {'Offsets':<8}")
    print("-" * 90)
    for r in suite_results:
        print(f"{r['id']:<6} | {r['name']:<35} | {r['prediction']:<12} | {r['latency_ms']:>7.1f} ms | {r['discrepancy']:>12.10f} | {r['offset_mismatches']:>7} err")
    print("=" * 90)

if __name__ == "__main__":
    run_test_suite()
