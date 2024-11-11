import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from googleapiclient.discovery import build 
import os
from dotenv import load_dotenv
from PIL import Image
from urllib.parse import urlparse, parse_qs
import re
from PIL import Image
from urllib.parse import urlparse, parse_qs
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from googletrans import Translator  # Import the Translator

def get_project_root():
    """Gets the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

# Load environment variables
load_dotenv(os.path.join(get_project_root(), '.env'))

# Get API key and validate
api_key = os.getenv("YOUTUBE_API_KEY")
if api_key is None:
    raise ValueError("API key is missing. Please set the YOUTUBE_API_KEY environment variable.")

# Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=api_key)

def load_css(file_name):
    project_root = get_project_root()
    css_path = os.path.join(project_root, 'screens', 'styles', file_name)
    
    try:
        with open(css_path) as f:
            css_content = f.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found at {css_path}")

def load_image(image_name):
    """Loads an image from the 'screens/images' folder."""
    project_root = get_project_root()
    image_path = os.path.join(project_root, 'screens', 'images', image_name)
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        st.error(f"Image '{image_name}' not found in the expected path.")
        return None

def extract_video_id(youtube_url):
    """Extracts video ID from various forms of YouTube URLs."""
    # Try parsing URL
    parsed_url = urlparse(youtube_url)

    if 'youtube.com' in parsed_url.netloc:
        if 'watch' in parsed_url.path:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif 'embed' in parsed_url.path:
            return parsed_url.path.split('/')[-1]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path[1:]
    
    video_id_pattern = r'^[A-Za-z0-9_-]{11}$'
    if re.match(video_id_pattern, youtube_url):
        return youtube_url
    
    return None

def translate_text(text):
    """Translates input text to English using googletrans."""
    translator = Translator()
    detected_language = translator.detect(text).lang
    if detected_language != 'en':
        translated = translator.translate(text, dest='en')  # Translate to English
        return translated.text, detected_language
    else:
        return text, detected_language
    
def preprocess_text(text):
    """Preprocesses text using the same steps as training data."""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def get_comments(video_id, max_results=50):
    """Fetches and translates comments from a YouTube video."""
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()
        
        comments = []
        for item in response.get('items', []):
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            translated_text, detected_language = translate_text(comment_text)  # Unpack the returned tuple
            processed_text = preprocess_text(translated_text)  # Preprocess the translated text
            comments.append({
                'processed_text': processed_text,
                'detected_language': detected_language  # Store detected language along with the processed text
            })
            
        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return []


def load_model():
    """Loads the BERT model for text classification."""
    try:
        project_root = get_project_root()
        model_path = os.path.join(project_root, 'models', 'fine_tuned_hate_speech_model')
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_prediction_message(label, score):
    """Converts label and score into a meaningful message."""
    if label == 'LABEL_0':
        category = 'Non-toxic'
        message = "This comment appears to be non-toxic and appropriate"
    else:
        category = 'Toxic'
        message = "This comment may contain toxic or inappropriate content"
    confidence_level = "high" if score > 0.8 else "moderate" if score > 0.6 else "low"
    return {
        'category': category,
        'message': message,
        'confidence_level': confidence_level
    }
def classify_comment(comment_info, classifier):
    """Classifies a single comment using the loaded model."""
    try:
        processed_comment = comment_info['processed_text']  # Get processed text from comment_info
        detected_language = comment_info['detected_language']  # Get detected language from comment_info
        prediction = classifier(processed_comment)[0]
        score = round(prediction['score'], 2)
        prediction_info = get_prediction_message(prediction['label'], score)
        
        # Return all relevant information including detected language
        return {
            'label': prediction_info['category'],
            'message': prediction_info['message'],
            'confidence_level': prediction_info['confidence_level'],
            'score': score,
            'processed_text': processed_comment,
            'detected_language': detected_language
        }
    except Exception as e:
        st.error(f"Error classifying comment: {str(e)}")
        return None


def display_prediction(prediction_info):
    if prediction_info is not None:
        st.markdown(f"### Prediction: **{prediction_info['label']}**", unsafe_allow_html=True)
        st.markdown(f"**Message**: {prediction_info['message']}", unsafe_allow_html=True)
        confidence_color = "green" if prediction_info['confidence_level'] == "high" else "orange" if prediction_info['confidence_level'] == "moderate" else "red"
        st.markdown(f"**Confidence Level**: <span style='color:{confidence_color};'>{prediction_info['confidence_level'].capitalize()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score**: {prediction_info['score']}", unsafe_allow_html=True)
        st.markdown(f"**Processed Comment**: {prediction_info['processed_text']}", unsafe_allow_html=True)
        st.markdown(f"**Original Language of the Comment**: <span style='color:green;'>{prediction_info['detected_language']}</span>", unsafe_allow_html=True)

def predict_text(text):
    """Predicts the classification for a single text input with translation and preprocessing."""
    try:
        # Translate the text
        translated_text, detected_language = translate_text(text)  # unpack the tuple

        # Now preprocess the translated text
        processed_text = preprocess_text(translated_text)
        
        # Load the model
        classifier = load_model()
        if classifier is None:
            return "Error: Model could not be loaded."

        result = classify_comment({'processed_text': processed_text, 'detected_language': detected_language}, classifier)
        if result is None:
            return "Error: Classification failed."
        
        # Display prediction results
        display_prediction(result)
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

    
def predict_youtube_comments(youtube_url):
    """Predicts classifications for comments from a YouTube video."""
    try:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return "Error: Invalid YouTube URL"
        
        classifier = load_model()
        if classifier is None:
            return "Error: Model could not be loaded."
        
        comments = get_comments(video_id)
        if not comments:
            return "Error: No comments found or could not fetch comments."
        
        for comment_info in comments:
            prediction = classify_comment(comment_info, classifier)
            if prediction:
                display_prediction(prediction)
                
    except Exception as e:
        return f"Error analyzing comments: {str(e)}"
