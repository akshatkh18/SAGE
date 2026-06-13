import pandas as pd
import json
from app.eda.profiler import profile_dataframe
from app.eda.insights import generate_insights, summarize_by_severity

df = pd.read_excel("D:/MLProjects/SAGE/online_retail_II.xlsx", sheet_name="Year 2009-2010")

profile = profile_dataframe(df)
insights = generate_insights(profile)
summary = summarize_by_severity(insights)

for severity in ["critical", "warning", "info"]:
    print(f"\n=== {severity.upper()} ({len(summary[severity])}) ===")
    for ins in summary[severity]:
        print(f"- [{ins['column']}] {ins['message']}")
        print(f"    -> {ins['suggested_action']}")