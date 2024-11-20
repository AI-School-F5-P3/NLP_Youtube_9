# firebase_utils.py

import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

class FirebaseManager:
    def __init__(self):
        # Initialize Firebase if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), 'secrets', 'serviceAccountKey.json'))
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    def save_prediction(self, prediction_data, source_info):
        try:
            # Add timestamp and source information
            document_data = {
                **prediction_data,
                'timestamp': datetime.now(),
                'source_type': source_info['type'],
                'source_content': source_info['content'],
                'source_id': source_info.get('video_id', None)  # For YouTube videos
            }
            
            # Add to predictions collection
            self.db.collection('predictions').add(document_data)
            return True
        except Exception as e:
            print(f"Error saving to Firebase: {str(e)}")
            return False
    
    def get_predictions(self, limit=100):
        try:
            predictions = self.db.collection('predictions')\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
            
            return [doc.to_dict() for doc in predictions]
        except Exception as e:
            print(f"Error retrieving from Firebase: {str(e)}")
            return []