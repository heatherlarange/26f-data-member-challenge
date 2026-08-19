# EDA on survey data as I build the models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("26f-data-member-challenge/data/survey.csv")
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(axis=1, how="all")

#  analyzing all possible outputs
print(df.isna().sum())
print(df["Age"].unique())
print(df["EdLevel"].unique())
print(df["OrgSize"].unique())
print(df["ICorPM"].unique())
print(df["RemoteWork"].unique())

# getting a scope of what salary distribution is/any outliers
print("\nSalary statistics:")
print(df["annual_salary_usd"].describe())

plt.plot(1, 2)
plt.hist(df["annual_salary_usd"], bins=75, edgecolor='black')
plt.xlabel('Salary (USD)')
plt.ylabel('Count')
plt.title('Salary Distribution Before Cleaning')

print(f"\nSalaries > $450k: {(df["annual_salary_usd"] > 400000).sum()}")
print(f"Salaries < $5k: {(df["annual_salary_usd"] < 5000).sum()}")
plt.figure(figsize=(12, 4))

# remove very small and large amounts
df = df[
    (df["annual_salary_usd"] >= 5000) &
    (df["annual_salary_usd"] <= 450000)
]

# show what current distribution looks like
print(f"\nAfter salary filtering:")
print(f"Rows: {len(df)}")
print(f"Salary range: ${df["annual_salary_usd"].min():,.0f} - ${df["annual_salary_usd"].max():,.0f}")
print(f"\nNew salary statistics:")
print(df["annual_salary_usd"].describe())

plt.subplot(1, 2, 1)
plt.hist(df["annual_salary_usd"], bins=50, edgecolor='black')
plt.xlabel('Salary (USD)')
plt.ylabel('Count')
plt.title('Salary Distribution After Cleaning')

# log shows more normal distribution for skewed data
plt.subplot(1, 2, 2)
plt.hist(np.log10(df["annual_salary_usd"]), bins=50, edgecolor='black')
plt.xlabel('Log10(Salary)')
plt.ylabel('Count')
plt.title('Log Salary Distribution')
plt.tight_layout()
plt.show()