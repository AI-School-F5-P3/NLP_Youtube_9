import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data.csv')
vector_path = os.path.join(data_dir, 'tfidf_vectorizer.joblib')

models_dir = os.path.join(current_dir, '..', 'models')
model_path = os.path.join(models_dir, 'nb_hatespeech_model.joblib')

df = pd.read_csv(data_path)

X = df.drop(columns=['IsHateSpeech'])
y = df['IsHateSpeech']

# Split the data into train and test sets
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'alpha': [0.1, 0.5, 1, 2, 5]
}

# Create the Naive Bayes model
model = MultinomialNB()

# Create the Grid Search object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')

# Fit the Grid Search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Hyperparameters: {best_params}")

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test F1-score: {test_f1:.2f}")

'''
# Save the best model
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")
'''