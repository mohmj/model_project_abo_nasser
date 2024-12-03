import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('synthetic_log_dataset.csv')

# Preprocessing
df['target'] = df['label']
X = df.drop(['target', 'label', 'action_no'], axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved successfully.")
