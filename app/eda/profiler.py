from ast import Dict
import pandas as pd 
import numpy as np

def profile_dataframe(df: pd.DataFrame)->Dict:
    # pass
    profile={}

    profile["n_rows"]=df.shape[0]
    profile["n_cols"]=df.shape[1]

    col_stats=[]
    for col in df.columns:
        stats={
            "name": col,
            "dtype": str(df[col].dtype),
            "missing": df[col].isnull().sum(), 
            "missing_pct": round(df[col].isnull().mean()*100, 2),
            "unique": df[col].nunique(),
            "unique_pct":  round(df[col].nunique()/len(df) *100,2)
        }

        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            stats['mean']= round(df[col].mean(),4), 
            stats['std']=round(df[col].std(), 4),
            stats['skew']=round(df[col].skew(), 4)
            stats['min']= df[col].min(),
            stats['max']= df[col].max(),
        col_stats.append(stats)

    profile['columns']=col_stats
    profile['duplicated_rows']=df.duplicated().sum()

    numeric_df=df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        profile['corr_matrix']=numeric_df.corr().round(3).to_dict()

    return profile