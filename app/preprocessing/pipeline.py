import pandas as pd 
import numpy as np 
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from app.eda.insights import NON_NEGATIVE_KEYWORDS
from app.utils.logger import logger 
from app.utils.exceptions import PreprocessingError

NON_NEGATIVE_KEYWORDS=[
    "quantity", "qty", "price", "cost", "amount", "amt", "salary",
    "age", "count", "units", "value", "revenue", "sales", "duration",
    "weight", "height", "distance", "score", "rating"
]

HIGH_CARDINALITY_THRESHOLD = 10
ID_COLUMN_THRESHOLD = 80

def log_action(report: list, column: str, action: str, reason: str):
    entry={"column": column, "action": action, "reason": reason}
    report.append(entry)
    logger.info(f"[Preprocessing] {column} | {action} | {reason}")

def _is_non_negative_column(col_name: str) -> bool:
    return any(kw in col_name.lower() for kw in NON_NEGATIVE_KEYWORDS)

def run_preprocessing(df: pd.DataFrame, profile: dict, target_column: str, high_cardinality_strategy: dict)->dict:
    try:
        logger.info(f"Starting preprocessing | shape={df.shape} | target='{target_column}'")

        df = df.copy()
        report=[]
        dropped_columns=[]

        col_profile= {c["name"]: c for c in profile["columns"]}

        for col in list(df.columns):
            if col==target_column:
                continue
            
            cprof = col_profile.get(col, {})
            dtype = cprof.get("dtype", str(df[col].dtype))

            unique_pct=cprof.get("unique_pct", 0)

            if unique_pct > ID_COLUMN_THRESHOLD and df[col].dtype != float:
                df.drop(columns=[col], inplace=True)
                dropped_columns.append(col)
                log_action(report, col, "Dropped",
                            f"High cardinality ID column ({unique_pct}% unique values).")
                continue

            if "datetime" in dtype:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                    df[f"{col}_hour"] = df[col].dt.hour
                    df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                    df.drop(columns=[col], inplace=True)
                    log_action(report, col, "Datetime extracted",
                                "Extracted year, month, day, dayofweek, hour, is_weekend features.")
                except Exception as e:
                    logger.warning(f"Datetime extraction failed for '{col}': {e}")
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                missing_pct = cprof.get("missing_pct", 0)
                outlier_pct = cprof.get("outlier_pct", 0)
                skew = cprof.get("skew", 0)
                negative_pct = cprof.get("negative_pct", 0)
                lower_bound = cprof.get("lower_bound", None)
                upper_bound = cprof.get("upper_bound", None)

                if missing_pct > 0:
                    median_val = df[col].median()
                    
                    if missing_pct > 5:
                        df[f"{col}_was_missing"] = df[col].isnull().astype(int)
                        log_action(report, f"{col}_was_missing", "Flag created",
                                    f"{missing_pct}% missing — created binary missingness indicator.")
                    
                    df[col].fillna(median_val, inplace=True)
                    
                    log_action(report, col, f"Imputed (median={round(median_val, 4)})",
                                f"{missing_pct}% missing values filled with median.")

                if negative_pct > 0 and _is_non_negative_column(col):
                    df[f"{col}_is_negative"] = (df[col] < 0).astype(int)
                    
                    log_action(report, f"{col}_is_negative", "Flag created",
                                f"{negative_pct}% negative values found in '{col}' — created binary flag.")

                if outlier_pct > 1.0 and lower_bound is not None and upper_bound is not None:
                    before = df[col].copy()
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    clipped = (before != df[col]).sum()
                    log_action(report, col, f"Winsorized ({clipped} values capped)",
                                f"{outlier_pct}% outliers detected (IQR method) — capped to [{round(lower_bound,2)}, {round(upper_bound,2)}].")

                elif abs(skew) > 1 and outlier_pct <= 1.0:
                    if df[col].min() >= 0:
                        df[col] = np.log1p(df[col])
                        log_action(report, col, "Log1p transform applied",
                                    f"Natural skew detected (skew={skew}) with low outlier rate — log1p applied.")
                    
                    else:
                        log_action(report, col, "Skew correction skipped",
                                    f"Skewed (skew={skew}) but column has negative values — log1p not safe.")
                                    
            elif dtype in ("object", "str") or pd.api.types.is_categorical_dtype(df[col]):

                missing_pct = cprof.get("missing_pct", 0)
                unique_count = cprof.get("unique", df[col].nunique())

                if missing_pct > 0:
                    df[col].fillna("Unknown", inplace=True)
                    
                    log_action(report, col, "Imputed (filled 'Unknown')",
                                f"{missing_pct}% missing values filled with 'Unknown'.")


                if unique_count > HIGH_CARDINALITY_THRESHOLD:
                    strategy = high_cardinality_strategy.get(col, "drop")

                    if strategy == "drop":
                        df.drop(columns=[col], inplace=True)
                        dropped_columns.append(col)
                        
                        log_action(report, col, "Dropped",
                                    f"High cardinality ({unique_count} unique) — user chose to drop.")
                        continue

                    elif strategy == "frequency_encode":
                        freq_map = df[col].value_counts(normalize=True).to_dict()
                        df[col] = df[col].map(freq_map)
                        
                        log_action(report, col, "Frequency encoded",
                                    f"High cardinality ({unique_count} unique) — replaced with frequency ratio.")

                    elif strategy == "target_encode":
                        if target_column in df.columns:
                            target_mean = df.groupby(col)[target_column].mean()
                            df[col] = df[col].map(target_mean)
                            
                            log_action(report, col, "Target encoded",
                                        f"High cardinality ({unique_count} unique) — replaced with target mean.")

                        else:
                            df.drop(columns=[col], inplace=True)
                            dropped_columns.append(col)
                            log_action(report, col, "Dropped (target encode failed)",
                                        "Target column not found for target encoding — dropped.")
                            continue

                    
                else:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    
                    log_action(report, col, f"One-Hot Encoded ({len(dummies.columns)} new cols)",
                                f"Low cardinality ({unique_count} unique) — OHE applied.")

        before=len(df)
        df.drop_duplicates(inplace=True)
        after=len(df)

        if before != after:
            log_action(report, "ALL", f"Dropped {before - after:,} duplicate rows",
                        "Duplicate rows removed before modeling.")

        remaining_nulls=df.isnull().sum().sum()
        if remaining_nulls>0:
            logger.warning(f"Preprocessing complete but {remaining_nulls} nulls remain. Applying fallback imputation.")
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            logger.info("Fallback imputation applied.")
        else:
            logger.info("Preprocessing complete — no nulls remaining.")

        feature_names = [c for c in df.columns if c != target_column]

        logger.info(f"Preprocessing done | output shape={df.shape} | features={len(feature_names)}")

        return {
            "df_processed": df,
            "preprocessing_report": report,
            "feature_names": feature_names,
            "dropped_columns": dropped_columns,
        }

    except PreprocessingError:
        raise
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise PreprocessingError(f"Preprocessing failed: {e}") from e
