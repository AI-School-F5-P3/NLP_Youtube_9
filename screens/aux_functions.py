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
    
    # Handle youtube.com URLs
    if 'youtube.com' in parsed_url.netloc:
        if 'watch' in parsed_url.path:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif 'embed' in parsed_url.path:
            return parsed_url.path.split('/')[-1]
    
    # Handle youtu.be URLs
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path[1:]
    
    # Try to extract video ID directly if it matches the pattern
    video_id_pattern = r'^[A-Za-z0-9_-]{11}$'
    if re.match(video_id_pattern, youtube_url):
        return youtube_url
    
    return None
def preprocess_text(text):
    """Preprocesses text using the same steps as training data."""
    # Convert to lowercase
    text = text.lower()

    text = re.sub(f"[{string.punctuation}]", "", text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

def get_comments(video_id, max_results=50):
    """Fetches comments from a YouTube video."""
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
            comments.append(comment_text)
            
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

def classify_comment(comment, classifier):
    """Classifies a single comment using the loaded model."""
    try:
        # Preprocess the comment before classification
        processed_comment = preprocess_text(comment)
        prediction = classifier(processed_comment)[0]
        score = round(prediction['score'], 4)
        prediction_info = get_prediction_message(prediction['label'], score)
        
        return {
            'label': prediction_info['category'],
            'message': prediction_info['message'],
            'confidence_level': prediction_info['confidence_level'],
            'score': score,
            'processed_text': processed_comment  # Add processed text to output
        }
    except Exception as e:
        st.error(f"Error classifying comment: {str(e)}")
        return None

def display_prediction(prediction_info):
    """Displays the prediction result in a pretty format in Streamlit."""
    if prediction_info is not None:
        # Displaying the prediction category
        st.markdown(f"### Prediction: **{prediction_info['label']}**", unsafe_allow_html=True)
        
        # Displaying the prediction message
        st.markdown(f"**Message**: {prediction_info['message']}", unsafe_allow_html=True)
        
        # Display confidence level in color
        if prediction_info['confidence_level'] == "high":
            confidence_color = "green"
        elif prediction_info['confidence_level'] == "moderate":
            confidence_color = "orange"
        else:
            confidence_color = "red"
        
        st.markdown(f"**Confidence Level**: <span style='color:{confidence_color};'>{prediction_info['confidence_level'].capitalize()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Score**: {prediction_info['score']}", unsafe_allow_html=True)
        st.markdown(f"**Processed Comment**: {prediction_info['processed_text']}", unsafe_allow_html=True)

def predict_text(text):
    """Predicts the classification for a single text input."""
    try:
        classifier = load_model()
        if classifier is None:
            return "Error: Model could not be loaded."
        
        result = classify_comment(text, classifier)
        if result is None:
            return "Error: Classification failed."
        
        # Now calling the function to display the prediction in a pretty format
        display_prediction(result)
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def predict_youtube_comments(youtube_url):
    """Predicts classifications for comments from a YouTube video."""
    try:
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        if not video_id:
            return "Error: Invalid YouTube URL"
        
        # Load model
        classifier = load_model()
        if classifier is None:
            return "Error: Model could not be loaded."
        
        # Get comments
        comments = get_comments(video_id)
        if not comments:
            return "Error: No comments found or could not fetch comments."
        
        for comment in comments:
            prediction = classify_comment(comment, classifier)
            if prediction:
             
                display_prediction(prediction)

    except Exception as e:
        return f"Error analyzing comments: {str(e)}"