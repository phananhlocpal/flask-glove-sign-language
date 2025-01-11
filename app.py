from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from typing import Dict, Any
import pandas as pd

app = Flask(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        
    def load_model(self, model_path: str) -> None:
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction using loaded model"""
        # Get model predictions
        predictions = self.model.predict(features)
        
        # Get predicted class and probability
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get top 3 predictions with their probabilities
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'label': int(idx),
                'probability': float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
        return {
            'predicted_label': int(predicted_class),
            'confidence': confidence,
            'top_3_predictions': top_3_predictions
        }

# Initialize model service
model_service = ModelService()

@app.before_first_request
def initialize_model():
    """Initialize model before first request"""
    try:
        # Update this path to your model location
        model_service.load_model('best_model.keras')
    except Exception as e:
        print(f"Error initializing model service: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_service.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get input data from request
        input_data = request.json
        
        # Validate input data
        if not input_data or not isinstance(input_data, list):
            return jsonify({'error': 'Input must be a list of features'}), 400
            
        # Convert input to numpy array
        features = np.array([input_data])
        
        # Make prediction
        prediction_result = model_service.predict(features)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)