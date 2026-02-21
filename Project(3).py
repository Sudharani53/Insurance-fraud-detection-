#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from ipywidgets import widgets, interactive, VBox, HBox, Layout
from IPython.display import display, clear_output
import datetime
import re


def get_dataset(fraud_type):
    if fraud_type == "Claim-based Fraud":
        data = {
            'Policy Tenure': [1,5,2,10,3,7,4,8,6,12],
            'Previous Claims': [0,1,0,2,1,3,0,2,1,4],
            'Incident Type': ['Accident','Fire','Theft','Accident','Fire','Theft','Accident','Fire','Theft','Accident'],
            'Claim Amount': [50000,200000,40000,180000,60000,250000,70000,150000,80000,300000],
            'Customer Age': [25,45,30,50,35,55,40,48,33,60],
            'Fraud': ['No','Yes','No','Yes','No','Yes','No','Yes','No','Yes']
        }
    else:
        data = {
            'Amount Paid to Officer': [0, 6000, 0, 8000, 2000, 10000, 0, 3000, 0, 7000],
            'Certificate Type': ['Medical','Accident','Fire','Vehicle','Other','Medical','Accident','Fire','Other','Vehicle'],
            'Previous Bribery': ['No','Yes','No','Yes','No','Yes','No','Yes','No','Yes'],
            'Claim Amount': [50000,200000,40000,180000,60000,250000,70000,150000,80000,300000],
            'Customer Age': [25,45,30,50,35,55,40,48,33,60],
            'Fraud': ['No','Yes','No','Yes','No','Yes','No','Yes','No','Yes']
        }
    return pd.DataFrame(data)


def train_model(fraud_type):
    df = get_dataset(fraud_type)
    X = df.drop('Fraud', axis=1)
    y = df['Fraud']

    le_dict = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    y = LabelEncoder().fit_transform(y)

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, le_dict, scaler, X.columns


def validate_inputs(inputs):
    errors = []
    if inputs['Email'] and not re.match(r"[^@]+@[^@]+\.[^@]+", inputs['Email']):
        errors.append("❌ Invalid email format")
    if inputs['Contact Number'] and (not str(inputs['Contact Number']).isdigit() or len(str(inputs['Contact Number'])) not in [10,11,12]):
        errors.append("❌ Invalid contact number")
    return errors


def predict_fraud(**user_inputs):
    clear_output(wait=True)
    display(ui)  # keep widgets visible
    errors = validate_inputs(user_inputs)
    if errors:
        for e in errors:
            print(e)
        return
    
    input_for_model = {k:v for k,v in user_inputs.items() if k not in ['Policy Holder Name','Policy ID','Contact Number','Email','Submission Date']}
    for col in le_dict.keys():
        try:
            input_for_model[col] = le_dict[col].transform([input_for_model[col]])[0]
        except:
            print(f"❌ Invalid value for {col}")
            return
    user_df = pd.DataFrame([input_for_model], columns=columns)
    imputer = SimpleImputer(strategy='median')
    user_df = pd.DataFrame(imputer.fit_transform(user_df), columns=columns)
    user_scaled = scaler.transform(user_df)
    pred = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]

    
    print("\n--- Prediction Results ---")
    print("User Inputs:")
    for k,v in user_inputs.items():
        print(f"{k}: {v}")
    print(f"\nPrediction: {'FRAUD 🚨' if pred==1 else 'NOT FRAUD ✅'}")
    print(f"Fraud Probability: {prob:.2f}")

    
    fig, ax = plt.subplots(figsize=(5,3))
    labels = ['Not Fraud','Fraud']
    values = [1-prob, prob]
    ax.bar(labels, values, color=['green','red'])
    ax.set_ylim(0,1)
    for i,vv in enumerate(values):
        ax.text(i, vv+0.02, f"{vv:.2f}", ha='center', fontweight='bold')
    ax.set_title("Fraud Probability")
    plt.show()

fraud_type_widget = widgets.Dropdown(
    options=["Claim-based Fraud","Bribery/Certificate Fraud"],
    description="Fraud Type:"
)
display(fraud_type_widget)


def on_type_change(change):
    global model, le_dict, scaler, columns
    model, le_dict, scaler, columns = train_model(change['new'])
fraud_type_widget.observe(on_type_change, names='value')
model, le_dict, scaler, columns = train_model(fraud_type_widget.value)


if fraud_type_widget.value == "Claim-based Fraud":
    user_inputs_widgets = {
        'Policy Tenure': widgets.IntText(value=1, description="Policy Tenure (Years)"),
        'Previous Claims': widgets.IntText(value=0, description="Previous Claims"),
        'Incident Type': widgets.Dropdown(options=["Accident","Fire","Theft"], description="Incident Type"),
        'Claim Amount': widgets.IntText(value=50000, description="Claim Amount (₹)"),
        'Customer Age': widgets.IntText(value=30, description="Customer Age"),
        'Policy Holder Name': widgets.Text(value="John Doe", description="Policy Holder Name"),
        'Policy ID': widgets.Text(value="POL123456", description="Policy ID"),
        'Contact Number': widgets.Text(value="9876543210", description="Contact Number"),
        'Email': widgets.Text(value="john@example.com", description="Email"),
        'Submission Date': widgets.DatePicker(value=datetime.date.today(), description="Submission Date")
    }
else:
    user_inputs_widgets = {
        'Amount Paid to Officer': widgets.IntText(value=0, description="Amount Paid to Officer (₹)"),
        'Certificate Type': widgets.Dropdown(options=["Medical","Accident","Fire","Vehicle","Other"], description="Certificate Type"),
        'Previous Bribery': widgets.Dropdown(options=["Yes","No"], description="Previous Bribery"),
        'Claim Amount': widgets.IntText(value=50000, description="Claim Amount (₹)"),
        'Customer Age': widgets.IntText(value=30, description="Customer Age"),
        'Policy Holder Name': widgets.Text(value="John Doe", description="Policy Holder Name"),
        'Policy ID': widgets.Text(value="POL123456", description="Policy ID"),
        'Contact Number': widgets.Text(value="9876543210", description="Contact Number"),
        'Email': widgets.Text(value="john@example.com", description="Email"),
        'Submission Date': widgets.DatePicker(value=datetime.date.today(), description="Submission Date")
    }

ui = VBox(list(user_inputs_widgets.values()))
display(ui)


button = widgets.Button(description="Check Fraud", button_style='danger')
out = widgets.Output()
def on_button_click(b):
    inputs = {k:widget.value for k,widget in user_inputs_widgets.items()}
    predict_fraud(**inputs)
button.on_click(on_button_click)
display(button, out)


# In[ ]:




