import pandas as pd
from data_sweep.profile import profile
from data_sweep.clean import clean
from data_sweep.report import report

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 4],
    "age": [25, 30, None, 40, 40],
    "flag": [1, 1, 1, 1, 1],
})

findings = profile(df)
print("--- FINDINGS ---")
for f in findings:
    print(f)

cleaned = clean(df)
print("\n--- CLEANED DATA ---")
print(cleaned)

summary = report(findings, df.shape, cleaned.shape)
print("\n--- REPORT ---")
print(summary)