import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, request, jsonify

# Configure TensorFlow to use only the CPU
# tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """Load the trained model and scaler"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully")
        except Exception as e:
            print(f"Error loading model or scaler: {str(e)}")
            raise
            
    def prepare_features(self, tilt: list, accel: list) -> np.ndarray:
        """Combine and preprocess tilt and accel data"""
        # Combine features in the same order as training
        features = np.concatenate([tilt, accel])
        features = features.reshape(1, -1)  # Reshape to 2D array for scaler
        
        # Apply the same scaling as during training
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
             
    def predict(self, features: np.ndarray) -> dict:
        """Make prediction using loaded model"""
        predictions = self.model.predict(features)
        predicted_class = np.argmax(predictions[0])
        
        return {
            'predicted_label': int(predicted_class)
        }
    

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize model service
    model_service = ModelService()

    try:
        # Update these paths to your model and scaler locations
        model_service.load_model(
            model_path='best_model.keras',
            scaler_path='scaler.joblib'
        )
    except Exception as e:
        print(f"Error initializing model service: {str(e)}")
        raise

    """Prediction endpoint"""
    try:
        # Get input data from request
        input_data = request.json
        
        # Validate input data
        if not input_data or 'tilt' not in input_data or 'accel' not in input_data:
            return jsonify({'error': 'Input must contain tilt and accel arrays'}), 400
            
        # Validate array lengths
        tilt = input_data['tilt']
        accel = input_data['accel']
        
        if len(tilt) != 11 or len(accel) != 9:  # Adjust these numbers based on your expected input dimensions
            return jsonify({'error': 'Invalid input dimensions. Expected tilt(11) and accel(9)'}), 400
        
        # Prepare and scale features
        features = model_service.prepare_features(tilt, accel)
        
        # Make prediction
        prediction_result = model_service.predict(features)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the model server!'})
