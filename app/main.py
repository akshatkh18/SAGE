import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import json

import pandas as pd
import seaborn as sns
import streamlit as st

import matplotlib.pyplot as plt

from app.eda.profiler import profile_dataframe
from app.eda.insights import generate_insights, summarize_by_severity
from app.eda.profiler import profile_dataframe
from app.utils.logger import logger
from app.utils.exceptions import DataLoadError, ProfilingError, InsightGenerationError

st.set_page_config(
    page_title="SAGE",
    page_icon="💕",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("💕SAGE💕")
    st.caption("Smart Adaptive Guided Experimentation")
    st.markdown("### Navigation")
    page = st.radio(
        "Navigation",
        ["EDA", "Preprocessing", "Model Arena", "Explainability", "Experiments", "Report"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("v1.0.0 | Zero-cost ML Co-pilot")

def load_data(uploaded_file)->pd.DataFrame:
    try:
        filename=uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df=pd.read_csv(uploaded_file)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            xl = pd.ExcelFile(uploaded_file)
            if len(xl.sheet_names) > 1:
                sheet = st.selectbox("Multiple sheets found. Select one:", xl.sheet_names)
            else:
                sheet = xl.sheet_names[0]
            df = pd.read_excel(uploaded_file, sheet_name=sheet)

        else:
            raise DataLoadError("Unsupported file type. Please upload a CSV or Excel file.")
        
        logger.info(f"File loaded: {uploaded_file.name} | Shape: {df.shape}")
        return df


    except DataLoadError:
        raise
    except Exception as e:
        logger.info(f"File load failed: {e}")
        raise DataLoadError(f"Could not load file: {e}") from e
        

def render_eda(df:pd.DataFrame):
    st.header("Exploratorary Data Analysis")

    st.subheader("Dataset Overview")
    col1, col2, col3, col4=st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.markdown("---")

    with st.spinner("Profiling Dataset...."):
        try:
            profile=profile_dataframe(df)
        except ProfilingError as e:
            st.error(f"Profiling failed: {e}")
            return 

    st.subheader("Column Statistics")
    col_data=[]
    for col in profile["columns"]:
        row = {
            "Column": col["name"],
            "Type": col["dtype"],
            "Missing %": col["missing_pct"],
            "Unique": col["unique"],
            "Unique %": col["unique_pct"],
        }
        if "mean" in col:
            row["Mean"] = col["mean"]
            row["Std"] = col["std"]
            row["Skew"] = col["skew"]
            row["Outlier %"] = col["outlier_pct"]
        col_data.append(row)

    st.dataframe(pd.DataFrame(col_data), use_container_width=True)

    st.markdown("---")

    st.subheader("Automated Insights")
    with st.spinner("Generating insights..."):
        try:
            insights = generate_insights(profile)
            summary = summarize_by_severity(insights)
        except InsightGenerationError as e:
            st.error(f"Insight generation failed: {e}")
            return

    severity_config = {
        "critical": ("🔴 Critical", "error"),
        "warning": ("🟡 Warning", "warning"),
        "info": ("🔵 Info", "info"),
    }

    for severity, (label, box_type) in severity_config.items():
        items = summary[severity]
        if not items:
            continue
        st.markdown(f"**{label} ({len(items)})**")
        for ins in items:
            col_tag = f"`{ins['column']}`" if ins["column"] else ""
            msg = f"{col_tag} {ins['message']}  \n**Action:** {ins['suggested_action']}"
            if box_type == "error":
                st.error(msg)
            elif box_type == "warning":
                st.warning(msg)
            else:
                st.info(msg)

    st.markdown("---")

    st.subheader("Visualizations")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlation Heatmap", "Missing Values"])

    with tab1:
        if numeric_cols:
            selected_col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax[0], color="#4F8BF9")
            ax[0].set_title(f"Distribution of {selected_col}")
            sns.boxplot(y=df[selected_col].dropna(), ax=ax[1], color="#4F8BF9")
            ax[1].set_title(f"Boxplot of {selected_col}")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No numeric columns found.")

    with tab2:
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                df[numeric_cols].corr(),
                annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5
            )
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Need at least 2 numeric columns for correlation heatmap.")

    with tab3:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            missing.plot(kind="bar", ax=ax, color="#4F8BF9")
            ax.set_title("Missing Values per Column")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()
        else:
            st.success("No missing values found in this dataset.")


def render_placeholder(name: str):
    st.header(name)
    st.info(f"{name} module is coming soon. Currently building Week 1 (EDA).")

def main():
    st.markdown("<h1 style='text-align:center'>🧠 SAGE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray'>Smart Adaptive Guided Experimentation — Your ML Co-pilot</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        st.toast("File received. Processing...", icon="📂")

    if uploaded_file is None:
        st.markdown("""
        ### Welcome to SAGE
        Upload a dataset to get started. SAGE will automatically:
        - Profile your data and detect quality issues
        - Generate intelligent, severity-ranked insights
        - Build an adaptive preprocessing pipeline
        - Train and compare multiple ML models
        - Explain predictions with SHAP
        - Generate a downloadable experiment report
        """)
        return

    with st.spinner("Reading file... this may take a moment for large files."):
        try:
            df = load_data(uploaded_file)
        except DataLoadError as e:
            st.error(str(e))
            return

    st.session_state["df"] = df
    st.session_state["filename"] = uploaded_file.name

    st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]:,} rows × {df.shape[1]} columns")

    if page == "EDA":
        with st.spinner("Running EDA pipeline..."):
            render_eda(df)

    elif page == "Preprocessing":
        if "df" not in st.session_state:
            st.warning("Please upload a dataset first from the EDA page.")
        else:
            from app.pages.preprocessing_page import render_preprocessing
            with st.spinner("Preparing preprocessing module..."):
                profile = profile_dataframe(st.session_state["df"])
            render_preprocessing(st.session_state["df"], profile)

    elif page == "Model Arena":
        render_placeholder("Model Arena")

    elif page == "Explainability":
        render_placeholder("Explainability")

    elif page == "Experiments":
        render_placeholder("Experiments")

    elif page == "Report":
        render_placeholder("Report")


if __name__ == "__main__":
    main()