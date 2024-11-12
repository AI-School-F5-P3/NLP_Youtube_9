<p align="center">
  <img src="screens/images/logo_bg.png" alt="Proyecto Logo" width="200"/>
</p>

# NLP Project - Group 9

## No Hate Zone: Using BERT, we analyze text to detect hate speech and potentially harmful content.

---

## Technologies Used in the Project

### 1. **User Interface**
   - **Streamlit**
   - 
### 2. **Natural Language Processing (NLP) and Machine Learning**
   - **Transformers (Hugging Face)**: Used to work with pre-trained language models, such as BERT, for text classification. Tools used include `BertTokenizer`, `BertForSequenceClassification` and `pipeline`.
   - **NLTK (Natural Language Toolkit)**
   - **Googletrans**
   - **Scikit-Learn**: Used for data preprocessing and evaluation metrics, such as `train_test_split`, `accuracy_score`, y `f1_score`.  Also uses `compute_class_weight` for balancing classes in the model.
   - **PyTorch**: Machine learning framework used for implementing and training neural networks, facilitating work with tensors and custom model definitions.
   - **MLflow**: Platform for managing the machine learning lifecycle, including metric logging and model persistence (`mlflow.pytorch`).

### 3. **Google Cloud Platform**
   - **Google API Client**: Connectivity and access to Google APIs, allowing interaction with other Google services, such as Google Cloud and **YouTube Data API**.
   - **Firebase**: Plataforma en la nube de Google para el desarrollo de aplicaciones. Componentes utilizados:
      - **Firebase Admin SDK**: Secure connection to Firebase from the backend, allowing access to Firestore.
      - **Firestore**: Real-time **NoSQL** database for storing and synchronizing structured application data, such as user information and prediction metrics.
        
---

## Deadlines

- **Final Submisson**: November 19, 2024

---

## Project Team

- **Esther Tapias**
- **Iryna Bilokon**
