import pandas as pd 
import numpy as np

from app.eda.profiler import profile_dataframe
from app.eda.insights import generate_insights

df =pd.read_excel("D:/MLProjects/SAGE/online_retail_II.xlsx")

# print(df.head())
print(df.shape)

profile = profile_dataframe(df)

insights = generate_insights(profile)
for i in insights:
    print("- " + i)

import json
print(json.dumps(profile, indent=2, default=str))
