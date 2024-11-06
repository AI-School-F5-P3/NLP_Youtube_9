from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data.csv')
vector_path = os.path.join(data_dir, 'tfidf_vectorizer.joblib')

models_dir = os.path.join(current_dir, '..', 'models')
model_path = os.path.join(models_dir, 'nb_hatespeech_model.joblib')

df = pd.read_csv(data_path)

X = df.drop(columns=['IsHateSpeech'])
y = df['IsHateSpeech']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

best_model = PassiveAggressiveClassifier(
    C=1, 
    max_iter=1000, 
    tol=0.0001, 
    early_stopping=False, 
    random_state=42
)

# Cross-validation on the training set
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

# Print the cross-validation results
print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation accuracy: {:.2f}".format(cv_scores.mean()))
print("Standard deviation: {:.2f}".format(cv_scores.std()))

# Train the model on the entire training set
best_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Overfitting check (comparing train and test accuracy)
train_accuracy = best_model.score(X_train, y_train)
print(f'Training accuracy: {train_accuracy:.2f}')
print(f'Test set accuracy: {test_accuracy:.2f}')

overfitting_percent = ((train_accuracy - test_accuracy) / train_accuracy) * 100
print(f'Overfitting percentage: {overfitting_percent:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Hate Speech', 'Hate Speech'])
disp.plot(cmap='Blues')

joblib.dump(best_model, model_path)

print(f'Model saved to {model_path}')