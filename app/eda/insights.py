from datetime import datetime
from enum import unique

import pandas as pd

from app.utils.logger import logger
from app.utils.exceptions import InsightGenerationError

NON_NEGATIVE_KEYWORDS = [
    "quantity", "qty", "price", "cost", "amount", "amt", "salary",
    "age", "count", "units", "value", "revenue", "sales", "duration",
    "weight", "height", "distance", "score", "rating"
]


def _make_insight(column, itype, severity, message, suggested_action):
    return {
        "column": column,
        "type": itype,
        "severity": severity,  
        "message": message,
        "suggested_action": suggested_action,
    }


def generate_insights(profile: dict) -> list[dict]:
    try:
        logger.info("Generating insights from profile")
        insights = []

        n_rows = profile["n_rows"]
        n_cols = profile["n_cols"]

        insights.append(_make_insight(
            column=None,
            itype="summary",
            severity="info",
            message=f"Dataset contains {n_rows:,} rows and {n_cols} columns.",
            suggested_action="None"
        ))

        dup = profile["duplicated_rows"]
        if dup > 0:
            dup_pct = round(dup / n_rows * 100, 2)
            severity = "warning" if dup_pct > 1 else "info"
            insights.append(_make_insight(
                column=None,
                itype="duplicates",
                severity=severity,
                message=f"Found {dup:,} duplicate rows ({dup_pct}% of data).",
                suggested_action="Drop duplicate rows before training."
            ))

        for col in profile["columns"]:
            name = col["name"]
            dtype = col["dtype"]
            missing_pct = col["missing_pct"]
            unique_pct = col["unique_pct"]
            unique_count = col["unique"]

            if missing_pct > 50:
                insights.append(_make_insight(
                    name, "missing_values", "critical",
                    f"Column '{name}' has {missing_pct}% missing values.",
                    "Consider dropping this column entirely."
                ))
            elif missing_pct > 20:
                insights.append(_make_insight(
                    name, "missing_values", "warning",
                    f"Column '{name}' has {missing_pct}% missing values.",
                    "Imputation recommended (median for numeric, mode for categorical)."
                ))
            elif 0 < missing_pct <= 20:
                insights.append(_make_insight(
                    name, "missing_values", "info",
                    f"Column '{name}' has {missing_pct}% missing values.",
                    "Safe to impute with median/mode."
                ))

            if unique_count == 1:
                insights.append(_make_insight(
                    name, "constant_column", "critical",
                    f"Column '{name}' has only 1 unique value.",
                    "Drop this column, it provides no information."
                ))

            if dtype == "object" and unique_pct > 80:
                insights.append(_make_insight(
                    name, "high_cardinality", "warning",
                    f"Column '{name}' has very high cardinality ({unique_count:,} unique values).",
                    "Likely an identifier column — exclude from modeling or use target encoding."
                ))

            if "skew" in col:
                skew = col["skew"]
                outlier_pct = col.get("outlier_pct", 0)

                if abs(skew) > 1:
                    if outlier_pct > 1.0:
                        insights.append(_make_insight(
                            name, "skew_outlier_driven", "warning",
                            f"Column '{name}' is highly skewed (skew={skew}), and {outlier_pct}% "
                            f"of values are outliers (IQR method).",
                            "Skew is likely outlier-driven — consider capping/winsorizing rather than a log transform."
                        ))
                    else:
                        insights.append(_make_insight(
                            name, "skew_natural", "info",
                            f"Column '{name}' is naturally skewed (skew={skew}), with low outlier rate ({outlier_pct}%).",
                            "Consider a log or power transform."
                        ))
                elif outlier_pct > 5:
                    insights.append(_make_insight(
                        name, "high_outlier_rate", "warning",
                        f"Column '{name}' has {outlier_pct}% outliers (IQR method) despite low skew.",
                        "Investigate outliers — may indicate data entry errors or a bimodal distribution."
                    ))

                neg_pct = col.get("negative_pct", 0)
                if neg_pct > 0:
                    name_lower = name.lower()
                    if any(kw in name_lower for kw in NON_NEGATIVE_KEYWORDS):
                        severity = "critical" if neg_pct > 10 else "warning"
                        insights.append(_make_insight(
                            name, "negative_values", severity,
                            f"Column '{name}' contains {col['negative_count']:,} negative values "
                            f"({neg_pct}%), which is unusual for a column with this name.",
                            "Investigate — may represent returns, refunds, or data entry errors. "
                            "Consider separating into a distinct flag/feature rather than dropping."
                        ))

                zero_pct = col.get("zero_pct", 0)
                if zero_pct > 5:
                    name_lower = name.lower()
                    if any(kw in name_lower for kw in ["price", "cost", "amount", "salary", "revenue"]):
                        insights.append(_make_insight(
                            name, "zero_values", "info",
                            f"Column '{name}' has {zero_pct}% zero values.",
                            "May indicate free items, promotions, or missing data encoded as zero — verify before modeling."
                        ))

            if "max_date" in col:
                try:
                    max_date = pd.to_datetime(col["max_date"])
                    min_date = pd.to_datetime(col["min_date"])
                    now = pd.Timestamp.now()

                    if max_date > now:
                        insights.append(_make_insight(
                            name, "future_dates", "warning",
                            f"Column '{name}' contains dates in the future (max: {max_date.date()}).",
                            "Verify data source — future timestamps may indicate logging errors or test data."
                        ))

                    if min_date.year < 1970:
                        insights.append(_make_insight(
                            name, "ancient_dates", "warning",
                            f"Column '{name}' contains suspiciously old dates (min: {min_date.date()}).",
                            "Check for placeholder/default date values (e.g. 1900-01-01)."
                        ))
                except Exception:
                    pass

        if "correlation_matrix" in profile:
            corr = profile["correlation_matrix"]
            cols = list(corr.keys())
            seen = set()
            for c1 in cols:
                for c2 in cols:
                    if c1 != c2 and (c2, c1) not in seen:
                        seen.add((c1, c2))
                        val = corr[c1][c2]
                        if abs(val) > 0.85:
                            insights.append(_make_insight(
                                f"{c1}, {c2}", "high_correlation", "warning",
                                f"Columns '{c1}' and '{c2}' are highly correlated ({val}).",
                                "Consider dropping one to reduce redundancy/multicollinearity."
                            ))
        logger.info(f"Generated {len(insights)} insights")

        return insights
    
    except Exception as e:
        logger.error(f"Insight generation failed: {e}")
        raise InsightGenerationError(f"Insight generation failed: {e}") from e


def summarize_by_severity(insights: list[dict]) -> dict:
    summary = {"critical": [], "warning": [], "info": []}
    for insight in insights:
        summary[insight["severity"]].append(insight)
    return summary