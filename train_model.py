import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
# Load cleaned data
df = pd.read_csv("cleaned_student.csv")
# Features and target
X = df[['hours_studied', 'attendance', 'previous_score', 'study_efficiency']]
y = df['final_score']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Choose model (try both)
model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = LinearRegression()
# Train
model.fit(X_train, y_train)
# Predict & evaluate
pred = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, pred))
print('MSE:', mean_squared_error(y_test, pred))
print('R2:', r2_score(y_test, pred))


# Save model
joblib.dump(model, 'model.pkl')
print('Model saved -> model.pkl')