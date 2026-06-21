import json

import pandas as pd
import seaborn as sns
import streamlit as st

import matplotlib.pyplot as plt

from app.eda.profiler import profile_dataframe
from app.eda.insights import generate_insights, summarize_by_severity
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
    st.markdown("---")
    st.markdown("###Navigation")
    page = st.radio("", ["EDA", "Preprocessing", "Model Arena", "Explainability", "Experiments", "Report"])
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
        
