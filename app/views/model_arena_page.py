import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from app.models.arena import run_model_arena, suggest_trials, estimate_time
from app.utils.logger import logger
from app.utils.exceptions import ModelTrainingError


def render_model_arena(df_processed: pd.DataFrame, target_column: str, feature_names: list):
    st.header("Model Arena")

    n_rows = len(df_processed)
    default_trials = suggest_trials(n_rows)

    st.subheader("Configuration")

    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.selectbox(
            "Optuna tuning trials",
            [10, 20, 30, 50, 100],
            index=[10, 20, 30, 50, 100].index(default_trials) if default_trials in [10, 20, 30, 50, 100] else 2
        )
    with col2:
        st.metric("Dataset Size", f"{n_rows:,} rows")

    est_time = estimate_time(n_rows, n_trials)
    st.caption(f"Estimated tuning time: {est_time} — base model training runs first and is usually under 10 seconds.")

    st.markdown("---")

    st.subheader("Models in Arena")
    cols = st.columns(7)
    for i, name in enumerate(["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "XGBoost", "LightGBM", "KNN"]):
        cols[i].markdown(f"<div style='text-align:center; padding:8px; background:#1e1e2e; border-radius:8px; font-size:12px'>{name}</div>", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Run Model Arena", type="primary", use_container_width=True):
        st.toast("Training all models...", icon="🏟️")

        st.markdown("**Phase 1: Training base models**")
        phase1_bar = st.progress(0, text="Starting...")

        st.markdown("**Phase 2: Optuna tuning best model**")
        phase2_bar = st.progress(0, text="Waiting for Phase 1...")

        def progress_callback(trial_num, total_trials, best_score):
            phase2_bar.progress(
                trial_num / total_trials,
                text=f"Trial {trial_num}/{total_trials} — Best F1/R²: {round(best_score, 4)}"
            )

        try:
            phase1_bar.progress(0.1, text="Training models...")

            with st.spinner("Running Model Arena — this may take a minute..."):
                arena_result = run_model_arena(
                    df_processed=df_processed,
                    target_column=target_column,
                    feature_names=feature_names,
                    n_trials=n_trials,
                    progress_callback=progress_callback
                )

            phase1_bar.progress(1.0, text="All base models trained!")
            phase2_bar.progress(1.0, text=f"Tuning complete!")

            st.session_state["arena_result"] = arena_result

            st.success("Model Arena complete!")
            st.markdown("---")

            st.subheader("Leaderboard")
            st.caption("Ranked by primary metric (F1 for classification, R² for regression)")

            leaderboard = arena_result["leaderboard"].copy()
            leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

            def highlight_best(row):
                if row["Rank"] == 1:
                    return ["background-color: #1a3a1a"] * len(row)
                return [""] * len(row)

            st.dataframe(
                leaderboard.style.apply(highlight_best, axis=1),
                use_container_width=True
            )

            st.markdown("---")

            st.subheader(f"Best Model: {arena_result['best_model_name']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Before Tuning**")
                for k, v in arena_result["untuned_metrics"].items():
                    st.metric(k, v)

            with col2:
                st.markdown("**After Optuna Tuning**")
                for k, v in arena_result["tuned_metrics"].items():
                    delta = round(v - arena_result["untuned_metrics"][k], 4)
                    st.metric(k, v, delta=delta)

            st.markdown("---")

            st.subheader("Best Hyperparameters")
            params_df = pd.DataFrame(
                arena_result["best_params"].items(),
                columns=["Parameter", "Value"]
            )
            st.dataframe(params_df, use_container_width=True)

            st.markdown("---")

            st.subheader("Model Comparison Chart")
            primary = "F1" if arena_result["task"] != "regression" else "R2"

            if primary in leaderboard.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ["#4F8BF9" if i > 0 else "#00cc66" for i in range(len(leaderboard))]
                ax.barh(leaderboard["Model"], leaderboard[primary], color=colors)
                ax.set_xlabel(primary)
                ax.set_title(f"Model Comparison by {primary}")
                ax.invert_yaxis()
                st.pyplot(fig)
                plt.close()

            if arena_result["trial_scores"]:
                st.subheader("Optuna Trial Progression")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(
                    range(1, len(arena_result["trial_scores"]) + 1),
                    arena_result["trial_scores"],
                    color="#4F8BF9", marker="o", markersize=3
                )
                ax.set_xlabel("Trial")
                ax.set_ylabel(f"Best {primary}")
                ax.set_title("Best Score per Optuna Trial")
                st.pyplot(fig)
                plt.close()

            st.success("Training complete. Head to **Explainability** for SHAP analysis.")
            logger.info(f"Model Arena UI complete | best={arena_result['best_model_name']}")

        except ModelTrainingError as e:
            st.error(f"Model Arena failed: {e}")
            logger.error(f"Model Arena UI error: {e}")