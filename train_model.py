import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data/churn.csv")

# Encode categorical columns
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Features & Target
X = df.drop(["CustomerID", "Churn"], axis=1)
y = df["Churn"]

# Train model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/churn_model.pkl")

print("✅ Model trained and saved successfully")
