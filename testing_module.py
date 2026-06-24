import pandas as pd
import json
from app.eda.profiler import profile_dataframe
from app.eda.insights import generate_insights, summarize_by_severity
from app.preprocessing.pipeline import run_preprocessing
from app.models.arena import run_model_arena, suggest_trials, estimate_time

# ─── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv("D:/MLProjects/SAGE/Churn_Modelling.csv")

# ─── Profiling ────────────────────────────────────────────────────────────────
profile = profile_dataframe(df)

# ─── Insights ─────────────────────────────────────────────────────────────────
insights = generate_insights(profile)
summary = summarize_by_severity(insights)

for severity in ["critical", "warning", "info"]:
    print(f"\n=== {severity.upper()} ({len(summary[severity])}) ===")
    for ins in summary[severity]:
        print(f"- [{ins['column']}] {ins['message']}")
        print(f"    -> {ins['suggested_action']}")

# ─── Preprocessing ────────────────────────────────────────────────────────────
high_card_strategy = {
    "Surname": "drop",
    "Geography": "frequency_encode",
    "Gender": "frequency_encode",
}

result = run_preprocessing(
    df=df,
    profile=profile,
    target_column="Exited",
    high_cardinality_strategy=high_card_strategy
)

df_processed = result["df_processed"]
print(f"\nProcessed shape: {df_processed.shape}")
print(f"Dropped columns: {result['dropped_columns']}")
print(f"Feature names: {result['feature_names']}")

print("\n=== Preprocessing Report ===")
for entry in result["preprocessing_report"]:
    print(f"[{entry['column']}] {entry['action']}")
    print(f"    Reason: {entry['reason']}")

# ─── Model Arena ──────────────────────────────────────────────────────────────
feature_names = result["feature_names"]

n_trials = suggest_trials(len(df_processed))
print(f"\nSuggested trials: {n_trials}")
print(f"Estimated time: {estimate_time(len(df_processed), n_trials)}")

arena_result = run_model_arena(
    df_processed=df_processed,
    target_column="Exited",
    feature_names=feature_names,
    n_trials=n_trials
)

print(f"\nTask: {arena_result['task']}")
print(f"\n=== Leaderboard ===")
print(arena_result["leaderboard"].to_string())
print(f"\nBest model: {arena_result['best_model_name']}")
print(f"Untuned metrics: {arena_result['untuned_metrics']}")
print(f"Tuned metrics: {arena_result['tuned_metrics']}")
print(f"Best params: {arena_result['best_params']}")
print(f"Model saved to: {arena_result['model_path']}")