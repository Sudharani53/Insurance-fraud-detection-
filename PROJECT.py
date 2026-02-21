#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("insurance_claims.csv")

# Drop useless columns
df = df.drop(['_c39', 'policy_number'], axis=1)

# Convert target
df['fraud_reported'] = df['fraud_reported'].map({'Y':1, 'N':0})

# -------------------------
# Convert dates FIRST
# -------------------------
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])

# Feature Engineering
df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days
df['claims_sum'] = df['injury_claim'] + df['property_claim'] + df['vehicle_claim']
df['claim_ratio'] = df['total_claim_amount'] / df['policy_annual_premium']
df['incident_weekday'] = df['incident_date'].dt.weekday

# Drop original date columns
df = df.drop(['policy_bind_date', 'incident_date'], axis=1)

# -------------------------
# Encode categorical columns
# -------------------------
from sklearn.preprocessing import LabelEncoder

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# -------------------------
# Split Data
# -------------------------
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Apply SMOTE ONLY (no class_weight)
# -------------------------
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------------
# Train Random Forest
# -------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_res, y_train_res)

# -------------------------
# Evaluation
# -------------------------
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

# -------------------------
# Threshold tuning
# -------------------------
threshold = 0.3
y_pred_thresh = (y_proba >= threshold).astype(int)

print("\nWith Threshold = 0.3")
print(confusion_matrix(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh))


# In[2]:


import pandas as pd

# Replace with your CSV file path
file_path = "insurance_claims.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Check the first few rows
print(df.head())

# Check column names and data types
print(df.info())
df = df.drop(['_c39', 'policy_number'], axis=1)
df['fraud_reported'] = df['fraud_reported'].map({'Y':1, 'N':0})
from sklearn.preprocessing import LabelEncoder

# Select only object-type columns
categorical_cols = df.select_dtypes(include='object').columns

# Encode them
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
	# Convert dates to datetime
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])

# Policy age in days
df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days

# Sum of claims
df['claims_sum'] = df['injury_claim'] + df['property_claim'] + df['vehicle_claim']

# Claim ratio to premium
df['claim_ratio'] = df['total_claim_amount'] / df['policy_annual_premium']

# Incident weekday (0=Monday, 6=Sunday)
df['incident_weekday'] = df['incident_date'].dt.weekday
df = df.drop(['policy_bind_date', 'incident_date'], axis=1)
# Target column
y = df['fraud_reported']

# Features
X = df.drop('fraud_reported', axis=1)
from sklearn.model_selection import train_test_split

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_proba))
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train model again on resampled data
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

# Predict on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
threshold = 0.3
y_pred_thresh = (y_proba >= threshold).astype(int)

print(confusion_matrix(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh))
import matplotlib.pyplot as plt

importances = model.feature_importances_
feat_names = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feat_names[indices], rotation=90)
plt.show()


# In[3]:


from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# -----------------------------
# Sample Training Dataset
# -----------------------------
data = pd.DataFrame({
    'claim_amount': [1000, 5000, 200, 7000, 300, 10000, 400, 15000],
    'customer_age': [25, 45, 22, 35, 28, 50, 30, 40],
    'previous_claims': [0, 2, 0, 3, 1, 4, 0, 5],
    'policy_duration': [12, 60, 6, 48, 12, 72, 24, 84],
    'fraud': [0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Genuine, 1 = Fraud
})

X = data[['claim_amount', 'customer_age', 'previous_claims', 'policy_duration']]
y = data['fraud']

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# Frontend HTML (Embedded)
# -----------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Insurance Fraud Detection</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f4f4f4;
            text-align: center;
            padding-top: 50px;
        }
        input {
            padding: 8px;
            margin: 5px;
            width: 200px;
        }
        button {
            padding: 10px 20px;
            background-color: green;
            color: white;
            border: none;
            cursor: pointer;
        }
        h2 {
            color: darkblue;
        }
    </style>
</head>
<body>

<h2>Insurance Fraud Detection System</h2>

<form method="POST" action="/predict">
    <input type="number" name="claim_amount" placeholder="Claim Amount" required><br>
    <input type="number" name="customer_age" placeholder="Customer Age" required><br>
    <input type="number" name="previous_claims" placeholder="Previous Claims" required><br>
    <input type="number" name="policy_duration" placeholder="Policy Duration (months)" required><br><br>
    
    <button type="submit">Check Fraud</button>
</form>

{% if prediction %}
    <h3>Prediction Result: {{ prediction }}</h3>
{% endif %}

</body>
</html>
"""

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    claim_amount = int(request.form['claim_amount'])
    customer_age = int(request.form['customer_age'])
    previous_claims = int(request.form['previous_claims'])
    policy_duration = int(request.form['policy_duration'])

    input_data = np.array([[claim_amount, customer_age, previous_claims, policy_duration]])

    prediction = model.predict(input_data)[0]
    result = "Fraud" if prediction == 1 else "Genuine"

    return render_template_string(HTML_PAGE, prediction=result)

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:




