from datetime import datetime
import os
import sys
import time

import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from app.utils.exceptions import ModelTrainingError
from app.utils.logger import logger 
optuna.logging.set_verbosity(optuna.logging.WARNING)

ARTIFACTS_DIR = "artifacts/models"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def detect_task(y:pd.Series)->str:
    unique=y.nunique()
    if unique==2:
        return "binary"
    elif unique<=10:
        return "multiclass"
    else:
        return "regression"

def suggest_trials(n_rows:int)->int:
    if n_rows <  5000:
        return 50
    elif n_rows < 20_000:
        return 30
    elif n_rows < 100_000:
        return 20
    else:
        return 10


def estimate_time(n_rows:int, n_trials:int)->str:
    seconds=int(n_rows * n_trials * 0.0002)
    if seconds<60:
        return f"~{seconds} seconds"
    else:
        return f"~{seconds // 60}m {seconds % 60}s"

def get_models(task:str)->dict:
    if task in ("binary", "multiclass"):
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
            "KNN": KNeighborsClassifier(),
        }

    else:
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(random_state=42, verbosity=0),
            "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
            "KNN": KNeighborsRegressor(),
        }

def evaluate_model(model, X_test, y_test, task:str)->dict:
    y_pred=model.predict(X_test)

    if task=="binary":
        y_prob=model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
        return {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred, average="binary"), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred, average="binary"), 4),
            "Recall": round(recall_score(y_test, y_pred, average="binary"), 4),
        }

    elif task=="multiclass":
        return {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred, average="binary"), 4),
            "Precision": round(precision_score(y_test, y_pred, average="binary"), 4),
            "Recall": round(recall_score(y_test, y_pred, average="binary"), 4),
        }

    else:
        return {
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "R2": round(r2_score(y_test, y_pred), 4),
        }


def get_optuna_objective(model_name: str,  task: str, X_train, y_train, X_test, y_test):
    primary_metric="F1" if task in ("binary", "multiclass") else "R2"

    def objective(trial):
        if model_name=="Random Forest":
            params={
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }

            model=RandomForestClassifier(**params, random_state=42) if task != "regression" else RandomForestRegressor(**params, random_state=42)
        
        elif model_name == "Extra Trees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
            }

            model = ExtraTreesClassifier(**params, random_state=42) if task != "regression" else ExtraTreesRegressor(**params, random_state=42)

        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

            model = XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0) if task != "regression" else XGBRegressor(**params, random_state=42, verbosity=0)

        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }

            model = LGBMClassifier(**params, random_state=42, verbose=-1) if task != "regression" else LGBMRegressor(**params, random_state=42, verbose=-1)

        elif model_name == "Logistic Regression":
            params = {
                "C": trial.suggest_float("C", 0.01, 10.0),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            }

            model = LogisticRegression(**params, max_iter=1000, random_state=42)

        elif model_name == "Decision Tree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }

            model = DecisionTreeClassifier(**params, random_state=42) if task != "regression" else DecisionTreeRegressor(**params, random_state=42)

        elif model_name == "KNN":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            }

            model = KNeighborsClassifier(**params) if task != "regression" else KNeighborsRegressor(**params)

        else:
            raise ValueError(f"No Optuna search space defined for {model_name}")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, task)
        return metrics[primary_metric]

    return objective

def run_model_arena(
    df_processed:  pd.DataFrame,
    target_column: str,
    feature_names: list,
    n_trials: int=20,
    progress_callback=None
)-> dict:
    try:
        logger.info(f"Model Arena started  | target='{target_column}' | trials={n_trials}")

        X = df_processed[feature_names].copy()
        y = df_processed[target_column].copy()

        # Clean y first — drop rows where target is missing
        if y.isnull().any():
            valid_idx = y.notnull()
            X = X[valid_idx].reset_index(drop=True)
            y = y[valid_idx].reset_index(drop=True)
            logger.warning(f"Dropped {(~valid_idx).sum()} rows with missing target values.")

        # Clean X — handle NaN and inf
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                median_val = X[col].median()
                if pd.isnull(median_val):
                    median_val = 0
                X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna("Unknown")

        # Final hard check
        remaining_nan = X.isnull().sum().sum()
        if remaining_nan > 0:
            bad_cols = X.isnull().sum()[X.isnull().sum() > 0].to_dict()
            logger.error(f"Still {remaining_nan} NaNs after cleaning: {bad_cols}")
            raise ModelTrainingError(f"Could not clean all NaN values in features: {bad_cols}")

        task = detect_task(y)
        logger.info(f"Task detected: {task}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task != "regression" else None
        )

        models=get_models(task)
        leaderboard=[]

        logger.info("Training Base Models")
        trained_models={}

        for name, model in models.items():
            start=time.time()
            model.fit(X_train, y_train)
            elapsed=round(time.time() - start, 2)
            metrics=evaluate_model(model, X_test, y_test, task)
            trained_models[name]=model

            row = {"Model": name, "Train Time (s)": elapsed, **metrics}
            leaderboard.append(row)
            logger.info(f"{name} trained in {elapsed}s | {metrics}")

        leaderboard_df = pd.DataFrame(leaderboard)
        primary_metric = "F1" if task in ("binary", "multiclass") else "R2"
        leaderboard_df = leaderboard_df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
        leaderboard_df.index += 1

        best_model_name = leaderboard_df.iloc[0]["Model"]
        best_model_untuned = trained_models[best_model_name]
        untuned_metrics = evaluate_model(best_model_untuned, X_test, y_test, task)

        logger.info(f"Best base model: {best_model_name} | {untuned_metrics}")

        logger.info(f"Phase 2: Optuna tuning {best_model_name} for {n_trials} trials")

        objective=get_optuna_objective(
            best_model_name, task, X_train, y_train, X_test, y_test
        )

        study=optuna.create_study(direction="maximize")

        trial_scores=[]

        def callback(study, trial):
            trial_scores.append(round(study.best_value, 4))
            if progress_callback:
                progress_callback(trial.number+1, n_trials, study.best_value)

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

        best_params=study.best_params
        objective_fn=get_optuna_objective(
            best_model_name, task, X_train, y_train, X_test, y_test
        )

        tuned_model=_build_model_from_params(best_model_name, best_params, task)

        tuned_model.fit(X_train, y_train)
        tuned_metrics=evaluate_model(tuned_model, X_test, y_test, task)

        logger.info(f"Tuned {best_model_name} | {tuned_metrics}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{best_model_name.replace(' ', '_')}_{timestamp}.pkl"
        model_path = os.path.join(ARTIFACTS_DIR, model_filename)
        joblib.dump(tuned_model, model_path)
        logger.info(f"Model saved to {model_path}")

        return {
            "task": task,
            "leaderboard": leaderboard_df,
            "best_model_name": best_model_name,
            "best_model_untuned": best_model_untuned,
            "best_model_tuned": tuned_model,
            "untuned_metrics": untuned_metrics,
            "tuned_metrics": tuned_metrics,
            "best_params": best_params,
            "trial_scores": trial_scores,
            "model_path": model_path,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
        }

    except ModelTrainingError:
        raise
    except Exception as e:
        logger.error(f"Model Arena Failed: {e}")
        raise ModelTrainingError(f"Model Arena failed: {e}") from e

def _build_model_from_params(model_name:str,  params:dict, task:str):
    is_clf =task != "regression"
    if model_name == "Random Forest":
        return RandomForestClassifier(**params, random_state=42) if is_clf else RandomForestRegressor(**params, random_state=42)
    elif model_name == "Extra Trees":
        return ExtraTreesClassifier(**params, random_state=42) if is_clf else ExtraTreesRegressor(**params, random_state=42)
    elif model_name == "XGBoost":
        return XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0) if is_clf else XGBRegressor(**params, random_state=42, verbosity=0)
    elif model_name == "LightGBM":
        return LGBMClassifier(**params, random_state=42, verbose=-1) if is_clf else LGBMRegressor(**params, random_state=42, verbose=-1)
    elif model_name == "Logistic Regression":
        return LogisticRegression(**params, max_iter=1000, random_state=42)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**params, random_state=42) if is_clf else DecisionTreeRegressor(**params, random_state=42)
    elif model_name == "KNN":
        return KNeighborsClassifier(**params) if is_clf else KNeighborsRegressor(**params)
    else:
        raise ModelTrainingError(f"Unknown model: {model_name}")
