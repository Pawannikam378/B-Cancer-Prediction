from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.svm import SVC
# Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state= 42)

# train model
model = SVC(kernel='linear', probability=True)

model.fit(X_train, y_train)

# Predict

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy: ', accuracy)

# Save model

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)