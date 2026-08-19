# analyzing all possible outputs
import pandas as pd
import numpy as np

df = pd.read_csv("26f-data-member-challenge/data/survey.csv")
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(axis=1, how="all")

print(df.isna().sum())
print(df["Age"].unique())
print(df["EdLevel"].unique())
print(df["OrgSize"].unique())
print(df["ICorPM"].unique())
print(df["RemoteWork"].unique())

print("\nSalary statistics:")
print(df["annual_salary_usd"].describe())

print(f"\nSalaries > $500k: {(df["annual_salary_usd"] > 500000).sum()}")
print(f"Salaries < $10k: {(df["annual_salary_usd"] < 10000).sum()}")
# add histograms

df = df[
    (df["annual_salary_usd"] >= 10000) &
    (df["annual_salary_usd"] <= 500000)
]
print(f"\nAfter salary filtering:")
print(f"Rows: {len(df)}")
print(f"Salary range: ${df["annual_salary_usd"].min():,.0f} - ${df["annual_salary_usd"].max():,.0f}")
print(f"\nNew salary statistics:")
print(df["annual_salary_usd"].describe())
# add histograms