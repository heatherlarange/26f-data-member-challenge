# first random forest model

from linear_regression import X_train_all, X_val_all, X_test_all, y_train, y_val, y_test
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# using random forest regressor
model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# use same preprocessed columns
model.fit(X_train_all, y_train)

val_pred = model.predict(X_val_all)

mse = mean_squared_error(y_val, val_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, val_pred)
r2 = r2_score(y_val, val_pred)

print("Performance on Validation Set:")
print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R²: {r2:.2f}")