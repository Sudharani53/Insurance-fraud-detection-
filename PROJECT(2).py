
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
def generate_synthetic_data(n_samples=500):
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'num_accidents': np.random.poisson(1, n_samples),
        'hospital_visits': np.random.poisson(2, n_samples),
        'claim_amount': np.random.randint(1000, 50000, n_samples),
        'policy_type': np.random.choice(['basic', 'premium', 'gold'], n_samples),
        'past_fraud_reports': np.random.randint(0, 3, n_samples),
        'days_since_last_claim': np.random.randint(0, 365, n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
        'employment_status': np.random.choice(['employed', 'unemployed', 'retired'], n_samples)
    })
    data = pd.get_dummies(data, columns=['policy_type', 'employment_status'], drop_first=True)

    # Synthetic fraud label
    data['fraud'] = np.where(
        (data['claim_amount'] > 30000) &
        (data['num_accidents'] > 1) &
        (data['past_fraud_reports'] > 0),
        1, 0
    )
    return data
def train_model(data):
    X = data.drop('fraud', axis=1)
    y = data['fraud']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns
def user_input_and_predict(model, feature_columns):
    print("\nEnter the details for insurance claim:")

    age = int(input("Age: "))
    num_accidents = int(input("Number of accidents: "))
    hospital_visits = int(input("Hospital visits: "))
    claim_amount = float(input("Claim amount: "))
    past_fraud_reports = int(input("Past fraud reports: "))
    days_since_last_claim = int(input("Days since last claim: "))
    num_dependents = int(input("Number of dependents: "))

    policy_type = input("Policy type (basic/premium/gold): ").lower()
    employment_status = input("Employment status (employed/unemployed/retired): ").lower()
    user_df = pd.DataFrame([[age, num_accidents, hospital_visits, claim_amount,
                             past_fraud_reports, days_since_last_claim, num_dependents]],
                           columns=['age', 'num_accidents', 'hospital_visits', 'claim_amount',
                                    'past_fraud_reports', 'days_since_last_claim', 'num_dependents'])
    for col in feature_columns:
        if col not in user_df.columns:
            user_df[col] = 0
    if policy_type == 'premium':
        user_df['policy_type_premium'] = 1
    elif policy_type == 'gold':
        user_df['policy_type_gold'] = 1
    if employment_status == 'unemployed':
        user_df['employment_status_unemployed'] = 1
    elif employment_status == 'retired':
        user_df['employment_status_retired'] = 1
    user_df = user_df[feature_columns]

    prediction = model.predict(user_df)
    result = "Fraudulent" if prediction[0] == 1 else "Genuine"
    print("\n--- Prediction Result ---")
    print(f"This insurance claim is: {result}")

if __name__ == "__main__":
    data = generate_synthetic_data()
    model, feature_columns = train_model(data)
    user_input_and_predict(model, feature_columns)


# In[ ]:




