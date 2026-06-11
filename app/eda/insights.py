from enum import unique


def generate_insights(profile: dict)->list[str]:
    insights=[]

    n_rows=profile["n_rows"]
    n_cols=profile["n_cols"]

    insights.append(f"Datatset conntain {n_rows} rows and {n_cols} columns.")

    dup=profile["duplicated_rows"]
    if dup>0:
        dup_pct=round(dup/n_rows *100, 2)
        insights.append(
            f"Found {dup:,} duplicate rows ({dup_pct}% of data). Now dropping these."
            )
    
    for col in profile["columns"]:
        name=col["name"]
        missing_pct=col["missing_pct"]
        unique_pct=col["unique_pct"]
        dtype=col["dtype"]

        if missing_pct > 50:
            insights.append(
                f"Column '{name}' has {missing_pct}% missing values"
                f"strongly consider dropping this column."
            )
        elif missing_pct>20:
            insights.append(
                f"Column '{name}' has {missing_pct}% missing values. "
                f"Imputation recommended (median/mode depending on type)."
            )

        elif missing_pct>=0 and missing_pct<=20:
            insights.append(
                f"Column '{name}' has {missing_pct}% missing values — "
                f"safe to impute."
            )

        if dtype=="object" and unique_pct>80:
            insights.append(
                f"Column '{name}' has very high cardinality ({col['unique']} unique values) — "
                f"likely an ID column, consider dropping or excluding from modeling."
            )

        if col["unique"]==1:
            insights.append(
                f"Column '{name}' has only 1 unique value — provides no information, drop it."
            )

        if "skew" in col:
            skew=col["skew"]
            if abs(skew)>1:
                direction="right" if skew>0 else "left"
                insights.append(
                    f"Column '{name}' is highly skewed ({direction}, skew={skew}) — "
                    f"consider a log or power transform."
                )

    if "corr_matrix" in profile:
        corr=profile["corr_matrix"]
        cols=list(corr.keys())
        seen=set()
        for c1 in cols:
            for c2 in cols:
                if c1!=c2 and (c2,c1) not in seen:
                    seen.add((c2,c2))
                    val=corr[c1][c2]
                    if abs(val)>0.85:
                        insights.append(
                        f"Columns '{c1}' and '{c2}' are highly correlated "
                        f"({val}) — consider dropping one to reduce redundancy."
                    )

    return insights