import pandas as pd 
import numpy as np

from app.eda.profiler import profile_dataframe

df =pd.read_excel("D:/MLProjects/SAGE/online_retail_II.xlsx")

# print(df.head())
print(df.shape)

profile = profile_dataframe(df)

import json
print(json.dumps(profile, indent=2, default=str))
