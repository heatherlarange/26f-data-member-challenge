# gradient model

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("26f-data-member-challenge/data/survey.csv")
df = df.replace("’", "'", regex=True)

# remove extremes
df = df[
    (df["annual_salary_usd"] >= 5000) &
    (df["annual_salary_usd"] <= 450000)
]

# remove na from salary column, filter unnecessary columns
df = df.dropna(subset=["annual_salary_usd"])
df = df.drop(columns=["ResponseId", "RemoteWork", "Currency"])

# convert age ordinal
age_map = {
    "18-24 years old": 1,
    "25-34 years old": 2,
    "35-44 years old": 3,
    "45-54 years old": 4,
    "55-64 years old": 5,
    "65 years or older": 6,
    "Prefer not to say": np.nan
}
df["Age"] = df["Age"].map(age_map)

# convert edlevel ordinal
education_map = {
    "Primary/elementary school": 1,
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
    "Some college/university study without earning a degree": 3,
    "Associate degree (A.A., A.S., etc.)": 4,
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 5,
    "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7,
    "Other (please specify):": np.nan
}
df["EdLevel"] = df["EdLevel"].map(education_map)

# convert orgsize ordinal
orgsize_map = {
    "Just me - I am a freelancer, sole proprietor, etc.": 1,
    "Less than 20 employees": 2,
    "20 to 99 employees": 3,
    "100 to 499 employees": 4,
    "500 to 999 employees": 5,
    "1,000 to 4,999 employees": 6,
    "5,000 to 9,999 employees": 7,
    "10,000 or more employees": 8,
    "I don't know": np.nan
}
df["OrgSize"] = df["OrgSize"].map(orgsize_map)

# convert ICorPM binary
icorpm_map = {
    "Individual contributor": 0,
    "People manager": 1
}
df["ICorPM"] = df["ICorPM"].map(icorpm_map)

# set variables, split dataset into training, validation, testing
X = df.drop(columns="annual_salary_usd")
y = df['annual_salary_usd'].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# preprocessing columns
numerical_columns = [
    "Age",
    "EdLevel",
    "WorkExp",
    "YearsCode",
    "OrgSize"
]

binary_columns = [
    "ICorPM"
]

categorical_columns = [
    "Employment",
    "DevType",
    "Industry",
    "Country",
]

multi_input_columns = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith"
]

# fill empty responses w median/mode/unknown/none
for col in numerical_columns:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_val[col] = X_val[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)

for col in binary_columns:
    mode = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode)
    X_val[col] = X_val[col].fillna(mode)
    X_test[col] = X_test[col].fillna(mode)

for col in categorical_columns:
    X_train[col] = X_train[col].fillna("Unknown")
    X_val[col] = X_val[col].fillna("Unknown")
    X_test[col] = X_test[col].fillna("Unknown")

for col in multi_input_columns:
    X_train[col] = X_train[col].fillna("None")
    X_val[col] = X_val[col].fillna("None")
    X_test[col] = X_test[col].fillna("None")

# use encoder for transforming categorical variables into numerical values
encoder = OneHotEncoder(
    handle_unknown="ignore",
    drop="first",
    sparse_output=True
)

X_train_categorical = encoder.fit_transform(X_train[categorical_columns])
X_val_categorical = encoder.transform(X_val[categorical_columns])
X_test_categorical = encoder.transform(X_test[categorical_columns])

# engineer the language/database columns into separate multi input columns
def create_multi_input_columns(train, val, test, column, prefix):
    train_values = train[column].fillna("None").str.split(";")
    val_values = val[column].fillna("None").str.split(";")
    test_values = test[column].fillna("None").str.split(";")
    categories = sorted(
        set(
            value.strip()
            for row in train_values
            for value in row
        )
    )
    train_output = pd.DataFrame(index=train.index)
    val_output = pd.DataFrame(index=val.index)
    test_output = pd.DataFrame(index=test.index)
    for category in categories:
        train_output[prefix + category] = train_values.apply(
            lambda x: int(
                category in [value.strip() for value in x]
            )
        )
        val_output[prefix + category] = val_values.apply(
            lambda x: int(
                category in [value.strip() for value in x]
            )
        )
        test_output[prefix + category] = test_values.apply(
            lambda x: int(
                category in [value.strip() for value in x]
            )
        )
    return train_output, val_output, test_output

train_languages, val_languages, test_languages = create_multi_input_columns(
    X_train,
    X_val,
    X_test,
    "LanguageHaveWorkedWith",
    "Language_"
)

train_databases, val_databases, test_databases = create_multi_input_columns(
    X_train,
    X_val,
    X_test,
    "DatabaseHaveWorkedWith",
    "Database_"
)

# give val/test data the same format as training, fill new/missing info with 0
val_languages = val_languages.reindex(
    columns=train_languages.columns,
    fill_value=0
)
val_databases = val_databases.reindex(
    columns=train_databases.columns,
    fill_value=0
)
test_languages = test_languages.reindex(
    columns=train_languages.columns,
    fill_value=0
)
test_databases = test_databases.reindex(
    columns=train_databases.columns,
    fill_value=0
)

X_train = X_train.drop(columns=["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith"])
X_val = X_val.drop(columns=["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith"])
X_test = X_test.drop(columns=["LanguageHaveWorkedWith", "DatabaseHaveWorkedWith"])

# combine numerical and binary columns
X_train_num = X_train[numerical_columns + binary_columns].values
X_val_num = X_val[numerical_columns + binary_columns].values
X_test_num = X_test[numerical_columns + binary_columns].values

# combine all columns into one train/val/test set
X_train_all = hstack([
    X_train_num,
    X_train_categorical,
    train_languages.values,
    train_databases.values
])

X_val_all = hstack([
    X_val_num,
    X_val_categorical,
    val_languages.values,
    val_databases.values
])

X_test_all = hstack([
    X_test_num,
    X_test_categorical,
    test_languages.values,
    test_databases.values
])

# convert values to dense
X_train_all = X_train_all.toarray()
X_val_all = X_val_all.toarray()
X_test_all = X_test_all.toarray()


# create models
models = {

    "GB 1": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    ),

    "GB 2": HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    ),

    "GB 3": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    ),

     "GB 4": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )
}

# validate different model types
for name, model in models.items():

    model.fit(X_train_all, y_train)

    y_val_pred = model.predict(X_val_all)

    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    print(name)
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.2f}")
    print()


# testing
for name, model in models.items():

    model.fit(X_train_all, y_train)

    y_pred = model.predict(X_test_all)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(name)
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.2f}")
    print()