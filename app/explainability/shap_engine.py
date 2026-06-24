import shap 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import os

from app.utils.logger import logger
from app.utils.exceptions import SAGEBaseException

ARTIFACTS_DIR = "artifacts/plots"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class ExplainabilityError(SAGEBaseException):
    pass

def get_explainer(model, X_train: pd.DataFrame):
    """
    Picks the right SHAP explainer based on model type.
    TreeExplainer for tree-based models, LinearExplainer for linear,
    KernelExplainer as fallback.
    """
    try:
        model_name=type(model).__name__
        logger.info(f"Creating SHAP explainer for {model_name}")

        tree_models=[
            "RandomForestClassifier", "RandomForestRegressor",
            "ExtraTreesClassifier", "ExtraTreesRegressor",
            "DecisionTreeClassifier", "DecisionTreeRegressor",
            "XGBClassifier", "XGBRegressor",
            "LGBMClassifier", "LGBMRegressor",
            "GradientBoostingClassifier", "GradientBoostingRegressor"
        ]

        linear_models=["LogisticRegression", "LinearRegression", "Ridge", "Lasso"]

        if model_name in tree_models:
            explainer=shap.TreeExplainer(model)
        elif model_name in linear_models:
            explainer = shap.LinearExplainer(model, X_train)
        else:
            background=shap.sample(X_train, 50)
            explainer=shap.KernelExplainer(model.predict_proba, background)

        logger.info(f"SHAP explainer created: {type(explainer).__name__}")
        return explainer

    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {e}")
        raise ExplainabilityError(f"Failed to create SHAP explainer: {e}") from e 

def compute_shap_values(explainer, X: pd.DataFrame):
    try:
        logger.info(f"Computing SHAP values for {len(X)} samples")
        shap_values = explainer(X)
        logger.info("SHAP values computed successfully")
        return shap_values
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        raise ExplainabilityError(f"SHAP computation failed: {e}") from e 

def plot_summary(shap_values, X: pd.DataFrame, save_path: str = None):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")
        return fig
    except Exception as e:
        logger.error(f"Summary plot failed: {e}")
        raise ExplainabilityError(f"Summary plot failed: {e}") from e

def plot_bar(shap_values, X: pd.DataFrame, save_path: str = None):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Bar plot saved to {save_path}")
        return fig
    except Exception as e:
        logger.error(f"Bar plot failed: {e}")
        raise ExplainabilityError(f"Bar plot failed: {e}") from e

def plot_waterfall(shap_values, row_idx: int, save_path: str = None):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[row_idx], show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Waterfall plot saved to {save_path}")
        return fig
    except Exception as e:
        logger.error(f"Waterfall plot failed: {e}")
        raise ExplainabilityError(f"Waterfall plot failed: {e}") from e

def generate_nl_explanation(shap_values, X: pd.DataFrame, row_idx: int, task: str) -> list[str]:
    try:
        if hasattr(shap_values, "values"):
            vals = shap_values[row_idx].values
        else:
            vals = shap_values[row_idx]

        if len(vals.shape) > 1:
            vals = vals[:, 1]

        feature_names = X.columns.tolist()
        row_values = X.iloc[row_idx].tolist()

        importance = sorted(
            zip(feature_names, vals, row_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanations = []

        for i, (feature, shap_val, actual_val) in enumerate(importance[:3]):
            direction = "increased" if shap_val > 0 else "decreased"
            impact = "strongly" if abs(shap_val) > 0.1 else "slightly"
            try:
                formatted_val = round(float(actual_val), 4)
            except (TypeError, ValueError):
                formatted_val = actual_val

            explanations.append(
                f"{i+1}. **{feature}** (value: `{formatted_val}`) "
                f"{impact} {direction} the prediction by `{round(abs(float(shap_val)), 4)}`."
            )

        positive_features = [f for f, v, _ in importance if v > 0]
        negative_features = [f for f, v, _ in importance if v < 0]

        if positive_features:
            explanations.append(
                f"\n**Pushing prediction higher:** {', '.join(positive_features[:3])}"
            )
        if negative_features:
            explanations.append(
                f"**Pushing prediction lower:** {', '.join(negative_features[:3])}"
            )

        return explanations

    except Exception as e:
        logger.error(f"NL explanation failed: {e}")
        return [f"Could not generate explanation: {e}"]


def get_feature_importance_df(shap_values, X: pd.DataFrame) -> pd.DataFrame:
    try:
        if hasattr(shap_values, "values"):
            vals = shap_values.values
        else:
            vals = shap_values

        if len(vals.shape) == 3:
            vals = vals[:, :, 1]

        mean_abs = np.abs(vals).mean(axis=0)
        df = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Mean |SHAP|": np.round(mean_abs, 4)
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        df.index += 1
        return df

    except Exception as e:
        logger.error(f"Feature importance DF failed: {e}")
        raise ExplainabilityError(f"Feature importance DF failed: {e}") from e

