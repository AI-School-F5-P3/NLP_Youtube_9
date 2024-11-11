import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Configurar los directorios
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data.csv')

models_dir = os.path.join(current_dir, '..', 'models', 'trained')
model_path = os.path.join(models_dir, 'xgb_hatespeech_model.joblib')
scaler_path = os.path.join(models_dir, 'xgb_hatespeech_scaler.joblib')

# Cargar los datos
df = pd.read_csv(data_path)

# Extraer características y la variable objetivo
X = df.drop(columns=['IsHateSpeech'])
y = df['IsHateSpeech']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el escalador para uso futuro
joblib.dump(scaler, scaler_path)
print(f"Scaler guardado en: {scaler_path}")

# Configurar el modelo de XGBoost con los hiperparámetros optimizados
xgb_model = xgb.XGBClassifier(
    n_estimators=213,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=1,
    subsample=0.95,
    colsample_bytree=0.78,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Entrenar el modelo
xgb_model.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = xgb_model.predict(X_test_scaled)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
joblib.dump(xgb_model, model_path)
print(f"Modelo guardado en: {model_path}")
