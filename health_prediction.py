import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset (Replace with your dataset)
df = pd.read_csv("health_data.csv")  # Ensure you have a dataset with symptoms and diseases

# Data Preprocessing
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])  # Encode target labels

X = df.drop(columns=['Disease'])  # Features
y = df['Disease']  # Target

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function for predicting disease
def predict_disease(symptoms):
    input_data = np.array(symptoms).reshape(1, -1)
    prediction = model.predict(input_data)
    disease = label_encoder.inverse_transform(prediction)[0]
    return disease

# Example Usage
sample_input = X_test.iloc[0].values  # Taking first test sample
predicted_disease = predict_disease(sample_input)
print(f"Predicted Disease: {predicted_disease}")
