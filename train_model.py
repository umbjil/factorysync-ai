import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'Type': np.random.choice([0, 1, 2], n_samples),  # 0=L, 1=M, 2=H
    'Air temperature [K]': np.random.normal(298.15, 5, n_samples),  # Mean 25°C
    'Process temperature [K]': np.random.normal(310.15, 8, n_samples),  # Mean 37°C
    'Rotational speed [rpm]': np.random.normal(6000, 1000, n_samples),
    'Torque [Nm]': np.random.normal(50, 10, n_samples),
    'Tool wear [min]': np.random.uniform(0, 200, n_samples)
}

# Generate target (failure probability increases with temperature and tool wear)
df = pd.DataFrame(data)
failure_prob = (
    0.3 * (df['Process temperature [K]'] > 315) +
    0.3 * (df['Tool wear [min]'] > 150) +
    0.2 * (df['Torque [Nm]'] > 60) +
    0.2 * (df['Rotational speed [rpm]'] > 7000)
)
df['Failure'] = np.random.binomial(1, failure_prob)

# Split features and target
X = df.drop('Failure', axis=1)
y = df['Failure']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Train accuracy: {train_score:.2%}")
print(f"Test accuracy: {test_score:.2%}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")
