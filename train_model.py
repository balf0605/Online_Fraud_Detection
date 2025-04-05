import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# Load the dataset
data = pd.read_csv("onlinefraud.csv")

# --- Step 1: Feature Engineering ---

# Balance differences
data['balance_diff_orig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['balance_diff_dest'] = data['oldbalanceDest'] - data['newbalanceDest']

# Time-based features
data['hour_of_day'] = data['step'] % 24
data['day_of_week'] = (data['step'] // 24) % 7

# Account behavior
data['is_orig_balance_zero'] = (data['oldbalanceOrg'] == 0).astype(int)
data['is_dest_balance_zero'] = (data['newbalanceDest'] == 0).astype(int)

# Relative amount (avoid division by zero)
data['amount_to_orig_balance'] = np.where(data['oldbalanceOrg'] > 0, data['amount'] / data['oldbalanceOrg'], 0)
data['amount_to_dest_balance'] = np.where(data['oldbalanceDest'] > 0, data['amount'] / data['oldbalanceDest'], 0)

# One-hot encode transaction type
data = pd.get_dummies(data, columns=['type'], prefix='is')

# Features to use (excluding identifiers like nameOrig, nameDest)
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'balance_diff_orig', 'balance_diff_dest', 'hour_of_day', 'day_of_week',
            'is_orig_balance_zero', 'is_dest_balance_zero', 'amount_to_orig_balance', 
            'amount_to_dest_balance'] + [col for col in data.columns if col.startswith('is_')]

# Target
y = data['isFraud']
X = data[features]

# --- Step 2: Exploratory Data Analysis (EDA) ---

# Fraud distribution
print("Fraud Distribution:\n", data['isFraud'].value_counts(normalize=True))

# Amount distribution by fraud
plt.figure(figsize=(10, 6))
sns.boxplot(x='isFraud', y='amount', data=data)
plt.title("Amount Distribution by Fraud Status")
plt.yscale('log')  # Log scale due to large range in amounts
plt.show()

# Fraud by hour
plt.figure(figsize=(12, 6))
sns.countplot(x='hour_of_day', hue='isFraud', data=data)
plt.title("Fraud by Hour of Day")
plt.show()

# --- Step 3: Train/Test Split ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Advanced Modeling with XGBoost ---

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# Train XGBoost model
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- Step 5: Model Evaluation ---

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Feature importance
feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 Feature Importance:\n", feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title("Top 10 Feature Importance")
plt.show()

# --- Step 6: Anomaly Detection with Isolation Forest ---

# Train on non-fraud data
non_fraud_data = data[data['isFraud'] == 0][features]
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(non_fraud_data)

# Predict anomalies on full dataset
data['anomaly_score'] = iso_forest.decision_function(X)
data['is_anomaly'] = iso_forest.predict(X)  # -1 for anomalies, 1 for normal

# Compare anomalies with fraud
anomaly_fraud_crosstab = pd.crosstab(data['is_anomaly'], data['isFraud'])
print("\nAnomaly vs Fraud Crosstab:\n", anomaly_fraud_crosstab)

# --- Step 7: Save Models ---

# Save XGBoost model
with open("xgboost_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Isolation Forest model
with open("isolation_forest_model.pkl", "wb") as file:
    pickle.dump(iso_forest, file)

print("\nModels saved as 'xgboost_model.pkl' and 'isolation_forest_model.pkl'.")