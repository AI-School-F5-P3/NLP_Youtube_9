import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data.csv')
vector_path = os.path.join(data_dir, 'tfidf_vectorizer.joblib')

models_dir = os.path.join(current_dir, '..', 'models', 'trained')
model_path = os.path.join(models_dir, 'nb_hatespeech_model.joblib')

df = pd.read_csv(data_path)

X = df.drop(columns=['IsHateSpeech'])
y = df['IsHateSpeech']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB(alpha=0.1)
# Train the model
model.fit(X_train, y_train)
'''
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
'''
# Get the training set performance
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Get the test set performance
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Calculate the overfitting
accuracy_diff = train_accuracy - test_accuracy
f1_diff = train_f1 - test_f1

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Accuracy Difference: {accuracy_diff:.2f}")

print(f"Training F1-score: {train_f1:.2f}")
print(f"Test F1-score: {test_f1:.2f}")
print(f"F1-score Difference: {f1_diff:.2f}")

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()