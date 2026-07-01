import streamlit as st
import os
from app.reports.generator import generate_report
from app.eda.insights import generate_insights
from app.explainability.shap_engine import get_feature_importance_df
from app.utils.logger import logger
from app.utils.exceptions import ReportGenerationError


def render_report(
    profile: dict,
    arena_result: dict,
    preprocessing_result: dict,
    dataset_name: str,
    target_column: str
):
    st.header("PDF Report Generator")
    st.caption("Generate a professional, shareable experiment report.")

    st.subheader("Report Contents")
    st.markdown("""
    The report will include:
    - **Cover page** with experiment metadata
    - **Dataset overview** — shape, column stats
    - **Automated insights** — severity-ranked findings
    - **Preprocessing pipeline** — all transformations with reasons
    - **Model Arena leaderboard** — all 7 models compared
    - **Tuning impact** — before vs after Optuna
    - **SHAP feature importance** — if Explainability was run
    - **Summary** — key findings in plain language
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", arena_result["best_model_name"])
    col2.metric("Task", arena_result["task"].capitalize())
    primary = "F1" if arena_result["task"] != "regression" else "R2"
    col3.metric(primary, arena_result["tuned_metrics"].get(primary, "N/A"))

    shap_available = "shap_values" in st.session_state
    if shap_available:
        st.success("SHAP values detected — feature importance will be included in the report.")
    else:
        st.info("SHAP values not found — run Explainability first to include feature importance.")

    st.markdown("---")

    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        st.toast("Building report...", icon="📄")
        with st.spinner("Generating PDF — this takes about 10 seconds..."):
            try:
                from app.eda.insights import generate_insights
                insights = generate_insights(profile)

                feature_importance_df = None
                if shap_available:
                    try:
                        shap_values = st.session_state["shap_values"]
                        X_test = st.session_state["shap_X_test"]
                        feature_importance_df = get_feature_importance_df(shap_values, X_test)
                    except Exception as e:
                        logger.warning(f"Could not extract SHAP importance for report: {e}")

                report_path = generate_report(
                    dataset_name=dataset_name,
                    target_column=target_column,
                    profile=profile,
                    insights=insights,
                    preprocessing_report=preprocessing_result.get("preprocessing_report", []),
                    dropped_columns=preprocessing_result.get("dropped_columns", []),
                    arena_result=arena_result,
                    feature_importance_df=feature_importance_df
                )

                st.success(f"Report generated!")

                with open(report_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                        use_container_width=True
                    )

                logger.info(f"Report page complete: {report_path}")

            except ReportGenerationError as e:
                st.error(f"Report generation failed: {e}")
                logger.error(f"Report page error: {e}")