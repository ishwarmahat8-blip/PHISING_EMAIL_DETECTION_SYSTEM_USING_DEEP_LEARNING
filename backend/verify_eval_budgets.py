import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent))
from explainers.shap_explainer import shap_explainer
from model_loader import model, tokenizer, device
import shap
import torch

class ModelCallCounter:
    """Wraps model forward passes to count exact invocations and batch sizes."""
    def __init__(self, original_model):
        self.model = original_model
        self.call_count = 0
        self.total_samples_evaluated = 0

    def reset(self):
        self.call_count = 0
        self.total_samples_evaluated = 0

def run_budget_analysis(email_title: str, subject: str, body: str, target_class_idx: int):
    print("=" * 90)
    print(f"ANALYSIS FOR: {email_title}")
    print(f"Target Class: {shap_explainer.class_names[target_class_idx]} (Index {target_class_idx})")
    print("=" * 90)

    text = f"Subject: {subject} \n Body: {body}"
    budgets = [100, 150, 200, 300, 500, 1000]

    budget_data = {}

    for b in budgets:
        # Instrumental wrapper to count model forward calls
        counter = {"forward_calls": 0, "total_samples": 0}

        def counting_predict_proba(texts):
            if isinstance(texts, str):
                texts = [texts]
            counter["forward_calls"] += 1
            counter["total_samples"] += len(texts)

            inputs = tokenizer(
                list(texts),
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
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs

        masker = shap.maskers.Text(tokenizer)
        t0 = time.perf_counter()
        explainer = shap.Explainer(counting_predict_proba, masker, output_names=["Legitimate", "Phishing"])
        shap_vals = explainer([text], max_evals=b)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Extract values
        base_val = float(shap_vals.base_values[0, target_class_idx])
        vals = shap_vals.values[0, :, target_class_idx]
        direct_prob = float(counting_predict_proba([text])[0, target_class_idx])
        sum_vals = float(np.sum(vals))
        discrepancy = abs((base_val + sum_vals) - direct_prob)

        # Explainer internal last eval count
        internal_eval_count = getattr(explainer, "last_eval_count", None)

        # Get features
        enc = tokenizer(text, return_offsets_mapping=True, max_length=256, truncation=True)
        raw_words = [tokenizer.decode([enc["input_ids"][i]]) for i in range(len(vals))]

        budget_data[b] = {
            "requested_max_evals": b,
            "actual_samples_evaluated": counter["total_samples"],
            "forward_calls_count": counter["forward_calls"],
            "internal_last_eval_count": internal_eval_count,
            "runtime_ms": elapsed_ms,
            "discrepancy": discrepancy,
            "values": vals,
            "words": raw_words,
            "base_val": base_val,
            "sum_vals": sum_vals
        }

    # Print Table 1: Model Evaluation Counts
    print("\n--- TABLE 1: Model Evaluation Counts and Latency ---")
    print(f"{'Req max_evals':<14} | {'Total Samples':<14} | {'Forward Calls':<14} | {'Runtime (ms)':<14} | {'Discrepancy':<16}")
    print("-" * 80)
    for b in budgets:
        d = budget_data[b]
        print(f"{d['requested_max_evals']:<14} | {d['actual_samples_evaluated']:<14} | {d['forward_calls_count']:<14} | {d['runtime_ms']:>12.1f} | {d['discrepancy']:>16.10f}")

    # Print Table 2: Consecutive Convergence Assessment
    print("\n--- TABLE 2: Consecutive Budget Convergence Metrics ---")
    print(f"{'Step (b1 -> b2)':<18} | {'Max Abs Diff':<13} | {'Mean Abs Diff':<14} | {'Top-10 Overlap':<15} | {'Sign Changes':<13} | {'Spearman rho':<12}")
    print("-" * 90)

    for i in range(len(budgets) - 1):
        b1 = budgets[i]
        b2 = budgets[i+1]
        v1 = budget_data[b1]["values"]
        v2 = budget_data[b2]["values"]

        # 1. Max and Mean diffs
        max_diff = float(np.max(np.abs(v2 - v1)))
        mean_diff = float(np.mean(np.abs(v2 - v1)))

        # 2. Top-10 features by absolute importance
        top10_idx1 = set(np.argsort(np.abs(v1))[-10:])
        top10_idx2 = set(np.argsort(np.abs(v2))[-10:])
        overlap_count = len(top10_idx1.intersection(top10_idx2))
        jaccard = overlap_count / len(top10_idx1.union(top10_idx2))

        # 3. Sign changes (excluding near-zero features < 1e-5)
        sign_changes = 0
        for j in range(len(v1)):
            if abs(v1[j]) > 1e-5 and abs(v2[j]) > 1e-5:
                if (v1[j] > 0 and v2[j] < 0) or (v1[j] < 0 and v2[j] > 0):
                    sign_changes += 1

        # 4. Spearman rank correlation
        rho, _ = spearmanr(v1, v2)

        print(f"{b1:>4d} -> {b2:<5d}      | {max_diff:>13.6f} | {mean_diff:>14.6f} | {overlap_count:>2d}/10 (J={jaccard:.2f}) | {sign_changes:>13d} | {rho:>12.5f}")

    return budget_data

if __name__ == "__main__":
    # Email 1: Long Corporate Email
    run_budget_analysis(
        email_title="Email 1: Long Corporate Policy Email (71 tokens)",
        subject="Enterprise Security Alert",
        body="To all staff members: Please review the updated corporate data protection protocols finalized yesterday. All workstations must adhere strictly to multi-factor authentication mandates, and database access logs will be audited bi-weekly by the internal governance team. If you observe any configuration discrepancies, submit a ticket immediately.",
        target_class_idx=0
    )

    # Email 2: High-Pressure Phishing Email
    run_budget_analysis(
        email_title="Email 2: High-Pressure Phishing Email (40 tokens)",
        subject="URGENT: Your account has been suspended! Verify now.",
        body="Dear customer, unauthorized login detected. Click here immediately to restore your account access or it will be permanently closed.",
        target_class_idx=1
    )

    # Email 3: Legitimate Meeting Email
    run_budget_analysis(
        email_title="Email 3: Legitimate Business Meeting Email (34 tokens)",
        subject="Project meeting scheduled for tomorrow at 10 AM",
        body="Hi team, please find attached the agenda for our weekly sprint review. Let me know if you have any questions.",
        target_class_idx=0
    )
