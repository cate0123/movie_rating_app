import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Dummy dataset for testing
data = {
    'budget': [100, 150, 200, 120, 300],
    'runtime': [90, 110, 150, 100, 130],
    'rating': [6.5, 7.0, 8.0, 6.8, 7.9]
}
df = pd.DataFrame(data)

X = df[['budget', 'runtime']]
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/movie_rating_model.pkl')

print("✅ Model trained and saved to model/movie_rating_model.pkl")
