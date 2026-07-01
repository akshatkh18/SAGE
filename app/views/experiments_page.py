import streamlit as st
import pandas as pd
import json
from app.storage.db import log_experiment, get_all_experiments, delete_experiment
from app.utils.logger import logger


def render_experiments(arena_result: dict, preprocessing_result: dict, dataset_name: str):
    st.header("Experiment Memory")
    st.caption("Every run you log is saved to SQLite and available for comparison.")

    st.subheader("Log Current Run")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", arena_result["best_model_name"])
    col2.metric("Task", arena_result["task"].capitalize())
    primary = "F1" if arena_result["task"] != "regression" else "R2"
    col3.metric(primary, arena_result["tuned_metrics"].get(primary, "N/A"))

    notes = st.text_input("Add notes for this run (optional)", placeholder="e.g. Churn dataset, dropped Surname, 30 Optuna trials")

    if st.button("Log This Experiment", type="primary", use_container_width=True):
        try:
            run_id = log_experiment(
                dataset_name=dataset_name,
                target_column=arena_result["feature_names"][0] if arena_result["feature_names"] else "unknown",
                task_type=arena_result["task"],
                best_model=arena_result["best_model_name"],
                n_features=len(arena_result["feature_names"]),
                n_rows=len(arena_result["X_test"]) * 5,  # approximate full dataset size
                untuned_metrics=arena_result["untuned_metrics"],
                tuned_metrics=arena_result["tuned_metrics"],
                best_params=arena_result["best_params"],
                dropped_columns=preprocessing_result.get("dropped_columns", []),
                preprocessing_steps=len(preprocessing_result.get("preprocessing_report", [])),
                model_path=arena_result["model_path"],
                notes=notes
            )
            st.success(f"Experiment logged with ID: `{run_id}`")
            logger.info(f"Experiment logged from UI: {run_id}")
        except Exception as e:
            st.error(f"Failed to log experiment: {e}")

    st.markdown("---")

    st.subheader("All Experiments")

    experiments = get_all_experiments()

    if not experiments:
        st.info("No experiments logged yet. Run the pipeline and log your first experiment above.")
        return

    rows = []
    for exp in experiments:
        tuned = exp.get("tuned_metrics", {})
        primary_val = tuned.get("F1") or tuned.get("R2") or "N/A"
        auc = tuned.get("AUC", "N/A")
        rows.append({
            "Run ID": exp["run_id"],
            "Timestamp": exp["timestamp"],
            "Dataset": exp["dataset_name"],
            "Target": exp["target_column"],
            "Task": exp["task_type"],
            "Best Model": exp["best_model"],
            "Features": exp["n_features"],
            primary: primary_val,
            "AUC": auc,
            "Notes": exp["notes"] or ""
        })

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Experiment Details")
    run_ids = [exp["run_id"] for exp in experiments]
    selected_run = st.selectbox("Select a run to inspect", run_ids)

    selected_exp = next(e for e in experiments if e["run_id"] == selected_run)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tuned Metrics**")
        for k, v in selected_exp["tuned_metrics"].items():
            st.metric(k, v)

    with col2:
        st.markdown("**Best Hyperparameters**")
        for k, v in selected_exp["best_params"].items():
            st.metric(k, round(float(v), 4) if isinstance(v, float) else v)

    with st.expander("Preprocessing info"):
        st.markdown(f"**Dropped columns:** {selected_exp['dropped_columns']}")
        st.markdown(f"**Preprocessing steps:** {selected_exp['preprocessing_steps']}")
        st.markdown(f"**Model path:** `{selected_exp['model_path']}`")

    st.markdown("---")
    if st.button(f"Delete `{selected_run}`", type="secondary"):
        delete_experiment(selected_run)
        st.warning(f"Experiment `{selected_run}` deleted.")
        st.rerun()

        