import pandas as pd

df = pd.read_csv("dataset/jobs.csv")

print(df["Benefits"].iloc[0])
print()
print(df["Company Profile"].iloc[0])