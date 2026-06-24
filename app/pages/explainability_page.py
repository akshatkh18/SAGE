import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from app.explainability.shap_engine import (
    get_explainer, compute_shap_values,
    plot_summary, plot_bar, plot_waterfall,
    generate_nl_explanation, get_feature_importance_df,
    ExplainabilityError
)
from app.utils.logger import logger

PLOT_DIR = "artifacts/plots"


def render_explainability(arena_result: dict):
    st.header("Explainability — SHAP Analysis")

    model = arena_result["best_model_tuned"]
    X_test = arena_result["X_test"]
    y_test = arena_result["y_test"]
    feature_names = arena_result["feature_names"]
    task = arena_result["task"]
    model_name = arena_result["best_model_name"]

    st.info(f"Explaining predictions from **{model_name}** (best tuned model)")

    if "shap_values" not in st.session_state:
        with st.spinner("Computing SHAP values — this may take 20-30 seconds..."):
            try:
                explainer = get_explainer(model, X_test)
                shap_values = compute_shap_values(explainer, X_test)
                st.session_state["shap_values"] = shap_values
                st.session_state["shap_X_test"] = X_test
                st.toast("SHAP values computed!", icon="✅")
            except ExplainabilityError as e:
                st.error(f"SHAP computation failed: {e}")
                return
    else:
        shap_values = st.session_state["shap_values"]
        X_test = st.session_state["shap_X_test"]

    st.success("SHAP values ready.")
    st.markdown("---")

    st.subheader("Global Feature Importance")
    st.caption("Ranked by mean absolute SHAP value across all test samples.")

    try:
        importance_df = get_feature_importance_df(shap_values, X_test)
        st.dataframe(importance_df, use_container_width=True)
    except ExplainabilityError as e:
        st.warning(f"Could not compute feature importance table: {e}")

    st.markdown("---")

    st.subheader("Global SHAP Plots")
    tab1, tab2 = st.tabs(["Beeswarm (Summary)", "Bar Chart"])

    with tab1:
        st.caption("Each dot is one test sample. Color = feature value. Position = SHAP impact.")
        with st.spinner("Generating summary plot..."):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"{PLOT_DIR}/summary_{timestamp}.png"
                fig = plot_summary(shap_values, X_test, save_path=save_path)
                st.pyplot(fig)
                plt.close()
            except ExplainabilityError as e:
                st.error(f"Summary plot failed: {e}")

    with tab2:
        st.caption("Mean absolute SHAP value per feature — cleaner view for non-technical audiences.")
        with st.spinner("Generating bar plot..."):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"{PLOT_DIR}/bar_{timestamp}.png"
                fig = plot_bar(shap_values, X_test, save_path=save_path)
                st.pyplot(fig)
                plt.close()
            except ExplainabilityError as e:
                st.error(f"Bar plot failed: {e}")

    st.markdown("---")

    st.subheader("Explain a Single Prediction")
    st.caption("Select a row from the test set to see why the model made that specific prediction.")

    col1, col2 = st.columns(2)
    with col1:
        row_idx = st.slider(
            "Select test row index",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0
        )
    with col2:
        actual = y_test.iloc[row_idx]
        predicted = model.predict(X_test.iloc[[row_idx]])[0]
        st.metric("Actual", int(actual))
        st.metric("Predicted", int(predicted),
                  delta="Correct ✓" if actual == predicted else "Wrong ✗",
                  delta_color="normal" if actual == predicted else "inverse")

    with st.expander("View row data"):
        st.dataframe(X_test.iloc[[row_idx]], use_container_width=True)

    with st.spinner("Generating waterfall plot..."):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{PLOT_DIR}/waterfall_{timestamp}.png"
            fig = plot_waterfall(shap_values, row_idx, save_path=save_path)
            st.pyplot(fig)
            plt.close()
        except ExplainabilityError as e:
            st.error(f"Waterfall plot failed: {e}")

    st.subheader("Natural Language Explanation")
    explanations = generate_nl_explanation(shap_values, X_test, row_idx, task)
    for line in explanations:
        st.markdown(line)

    st.markdown("---")
    st.info("Head to **Experiments** to log this run, or **Report** to generate a PDF.")
    logger.info(f"Explainability page rendered | row_idx={row_idx}")