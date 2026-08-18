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