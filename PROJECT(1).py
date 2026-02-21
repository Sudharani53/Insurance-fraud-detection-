#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
file_path = "insurance_claims.csv"
df = pd.read_csv(file_path)
print(df.head())
print(df.info())
df = df.drop(['_c39', 'policy_number'], axis=1)
df['fraud_reported'] = df['fraud_reported'].map({'Y':1, 'N':0})
from sklearn.preprocessing import LabelEncoder
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days
df['claims_sum'] = df['injury_claim'] + df['property_claim'] + df['vehicle_claim']
df['claim_ratio'] = df['total_claim_amount'] / df['policy_annual_premium']
df['incident_weekday'] = df['incident_date'].dt.weekday
df = df.drop(['policy_bind_date', 'incident_date'], axis=1)
y = df['fraud_reported']
X = df.drop('fraud_reported', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_proba))
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)
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


# In[ ]:





# In[ ]:




