import streamlit as st
import pandas as pd
from app.eda.profiler import profile_dataframe
from app.preprocessing.pipeline import run_preprocessing, HIGH_CARDINALITY_THRESHOLD
from app.utils.logger import logger
from app.utils.exceptions import PreprocessingError

def render_preprocessing(df: pd.DataFrame, profile: dict):
    st.header("Preprocessing Pipeline")

    st.subheader("Step 1: Select Target Column")
    all_columns = df.columns.tolist()
    target_column = st.selectbox(
        "Which column are you trying to predict?",
        all_columns,
        index=len(all_columns) - 1
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Type:** `{df[target_column].dtype}`")
        st.markdown(f"**Unique values:** {df[target_column].nunique()}")
    with col2:
        if df[target_column].nunique() <= 10:
            counts = df[target_column].value_counts()
            st.table({"Value": list(counts.index), "Count": list(counts.values)})

    st.markdown("---")

    st.subheader("Step 2: Handle High-Cardinality Columns")

    high_card_cols = []
    for col in profile["columns"]:
        if col["name"] == target_column:
            continue
        if col["dtype"] in ("object", "str") and col["unique"] > HIGH_CARDINALITY_THRESHOLD:
            high_card_cols.append(col)

    high_cardinality_strategy = {}

    if not high_card_cols:
        st.success("No high-cardinality categorical columns detected.")
    else:
        st.info(f"Found {len(high_card_cols)} high-cardinality column(s). Choose how SAGE should handle each:")
        for col in high_card_cols:
            name = col["name"]
            unique = col["unique"]
            unique_pct = col["unique_pct"]
            
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**{name}**")
                st.caption(f"{unique:,} unique values ({unique_pct}%)")
            with col2:
                strategy = st.selectbox(
                    f"Strategy for '{name}'",
                    ["drop", "frequency_encode", "target_encode"],
                    format_func=lambda x: {
                        "drop": "Drop column (recommended for IDs)",
                        "frequency_encode": "Frequency encode (replace with frequency ratio)",
                        "target_encode": "Target encode (replace with target mean)"
                    }[x],
                    key=f"strategy_{name}"
                )
                high_cardinality_strategy[name] = strategy

    st.markdown("---")

    st.subheader("Step 3: Run Preprocessing")

    col1, col2, col3 = st.columns(3)
    col1.metric("Input Shape", f"{df.shape[0]:,} × {df.shape[1]}")
    col2.metric("Target Column", target_column)
    col3.metric("High-Card Columns", len(high_card_cols))

    if st.button("Run Preprocessing", type="primary", use_container_width=True):
        st.toast("Pipeline started...", icon="⚙️")
        with st.spinner("Running adaptive preprocessing pipeline..."):
            try:
                result = run_preprocessing(
                    df=df,
                    profile=profile,
                    target_column=target_column,
                    high_cardinality_strategy=high_cardinality_strategy
                )

                st.session_state["df_processed"] = result["df_processed"]
                st.session_state["target_column"] = target_column
                st.session_state["feature_names"] = result["feature_names"]
                st.session_state["preprocessing_report"] = result["preprocessing_report"]
                st.session_state["dropped_columns"] = result["dropped_columns"]

                df_processed = result["df_processed"]
                report = result["preprocessing_report"]
                dropped = result["dropped_columns"]
 
                st.success("Preprocessing complete!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Output Shape", f"{df_processed.shape[0]:,} × {df_processed.shape[1]}")
                col2.metric("Features Created", len(result["feature_names"]))
                col3.metric("Columns Dropped", len(dropped))

                st.markdown("---")

                st.subheader("Preprocessing Report")
                st.caption("Here's exactly what SAGE did to your data and why.")

                report_df = pd.DataFrame(report)
                
                for _, row in report_df.iterrows():
                    with st.expander(f"[{row['column']}] — {row['action']}"):
                        st.markdown(f"**Reason:** {row['reason']}")

                st.markdown("---")

                st.subheader("Processed Data Preview")
                st.dataframe(df_processed.head(10), use_container_width=True)

                with st.expander("View all feature names"):
                    st.write(result["feature_names"])

                st.info("Preprocessed data saved. Head to **Model Arena** to train models.")
                logger.info(f"Preprocessing UI complete | output={df_processed.shape}")

            except PreprocessingError as e:
                st.error(f"Preprocessing failed: {e}")
                logger.error(f"Preprocessing UI error: {e}")