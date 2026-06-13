from ast import Dict
import pandas as pd 
import numpy as np

def profile_dataframe(df: pd.DataFrame)->Dict:
    profile={}

    profile["n_rows"]=df.shape[0]
    profile["n_cols"]=df.shape[1]

    col_stats=[]
    for col in df.columns:
        series = df[col]
        stats={
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": df[col].isnull().sum(), 
            "missing_pct": round(df[col].isnull().mean()*100, 2),
            "unique": df[col].nunique(),
            "unique_pct":  round(df[col].nunique()/len(df) *100,2)
        }



        if pd.api.types.is_numeric_dtype(series):
            clean=series.dropna()
            stats['mean']= round(clean.mean(),4)
            stats['std']=round(clean.std(), 4)
            stats['skew']=round(clean.skew(), 4)
            stats['min']= clean.min()
            stats['max']= clean.max()

            q1=clean.quartile(0.25)
            q3=clean.quartile(0.75)
            iqr=q3-q1
            lower=q1-1.5*iqr
            upper=q3+1.5*iqr
            outliers=clean[(clean < lower) | (clean >upper)]

            stats["q1"]=round(q1,4)
            stats["q3"]=round(q3,4)
            stats["iqr"]=round(iqr,4)
            stats["lower_bound"]= round(lower,4)
            stats["upper_bound"]= round(upper, 4)
            stats["outliers_count"]=int(len(outliers))
            stats["outlier_pct"] = round(len(outliers) / len(clean) * 100, 2) if len(clean) > 0 else 0.0

            stats["negative_count"]=int((clean<0).sum())
            stats["negative_pct"]=round((clean<0).mean()*100, 2)
            stats["zero_count"]=int((clean==0).sum())
            stats["zero_pct"]=round((clean==0).mean()*100, 2)

        if pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            if not clean.empty:
                stats["min_date"] = str(clean.min())
                stats["max_date"] = str(clean.max())

        col_stats.append(stats)

    profile['columns']=col_stats
    profile['duplicated_rows']=int(df.duplicated().sum())

    numeric_df=df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        profile['corr_matrix']=numeric_df.corr().round(3).to_dict()

    return profile
    