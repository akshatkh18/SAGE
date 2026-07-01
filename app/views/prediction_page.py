import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from app.explainability.shap_engine import (
    get_explainer, compute_shap_values,
    plot_waterfall, generate_nl_explanation,
    ExplainabilityError
)
from app.utils.logger import logger


def render_prediction(arena_result: dict, df_processed: pd.DataFrame):
    st.header("Real-Time Prediction")
    st.caption("Enter feature values manually and SAGE will predict and explain the outcome.")

    model = arena_result["best_model_tuned"]
    feature_names = arena_result["feature_names"]
    task = arena_result["task"]
    model_name = arena_result["best_model_name"]

    st.info(f"Using **{model_name}** (best tuned model)")
    st.markdown("---")

    st.subheader("Input Feature Values")

    X_test = arena_result["X_test"]

    input_values = {}
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        col = cols[i % 3]
        with col:
            series = X_test[feature]
            unique_vals = series.nunique()
            dtype = series.dtype

            if unique_vals <= 2:
                val = st.selectbox(
                    feature,
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    key=f"input_{feature}"
                )
            elif unique_vals <= 10 and dtype in [np.int64, np.int32]:
                val = st.selectbox(
                    feature,
                    options=sorted(series.unique().tolist()),
                    key=f"input_{feature}"
                )
            else:
                min_val = float(series.min())
                max_val = float(series.max())
                mean_val = float(series.mean())
                slider_val = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"slider_{feature}"
                )
                val = st.number_input(
                    f"{feature} (type exact value)",
                    min_value=min_val,
                    max_value=max_val,
                    value=slider_val,
                    key=f"input_{feature}"
                )
            input_values[feature] = val

    st.markdown("---")

    if st.button("Predict", type="primary", use_container_width=True):
        st.toast("Running prediction...", icon="🔮")

        try:
            input_df = pd.DataFrame([input_values])

            prediction = model.predict(input_df)[0]
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                confidence = round(float(max(proba)) * 100, 2)
                churn_prob = round(float(proba[1]) * 100, 2) if task == "binary" else None
            else:
                confidence = None
                churn_prob = None

            st.markdown("---")
            st.subheader("Prediction Result")

            # Result display
            col1, col2, col3 = st.columns(3)
            
            if task == "binary":
                result_label = "YES" if prediction == 1 else "NO"
                result_color = "🔴" if prediction == 1 else "🟢"
                col1.metric("Prediction", f"{result_color} {result_label}")
                if churn_prob is not None:
                    col2.metric("Churn Probability", f"{churn_prob}%")
                if confidence is not None:
                    col3.metric("Confidence", f"{confidence}%")
            elif task == "multiclass":
                col1.metric("Predicted Class", str(prediction))
                if confidence is not None:
                    col2.metric("Confidence", f"{confidence}%")
            else:
                col1.metric("Predicted Value", round(float(prediction), 4))

            st.markdown("---")

            st.subheader("Why this prediction?")

            with st.spinner("Computing SHAP explanation..."):
                try:
                    explainer = get_explainer(model, X_test)
                    shap_values = compute_shap_values(explainer, input_df)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0], show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.subheader("Plain English Explanation")
                    explanations = generate_nl_explanation(
                        shap_values, input_df, 0, task
                    )
                    for line in explanations:
                        st.markdown(line)

                except ExplainabilityError as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

            logger.info(f"Prediction made: {prediction} | confidence: {confidence}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.error(f"Prediction page error: {e}")