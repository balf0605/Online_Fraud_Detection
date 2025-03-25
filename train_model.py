import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("onlinefraud.csv")

# Map categorical values to numerical
data["type"] = data["type"].map({
    "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5
})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

# Prepare features and labels
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data["isFraud"])

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Save model and encoder
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("Model training complete and saved as 'model.pkl'.")
