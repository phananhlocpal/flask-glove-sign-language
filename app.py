import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from flask import Flask, request, jsonify

# Define SEBlock class
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = 16

    def build(self, input_shape):
        self.squeeze = tf.keras.layers.GlobalAveragePooling1D()
        self.excitation = tf.keras.Sequential([
            tf.keras.layers.Dense(input_shape[1] // self.reduction_ratio, activation='relu'),
            tf.keras.layers.Dense(input_shape[1], activation='sigmoid')
        ])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = self.squeeze(inputs)
        x = self.excitation(x)
        x = tf.reshape(x, [batch_size, 1, -1])
        return inputs * x

# Define AttentionBlock class
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = None

    def build(self, input_shape):
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(input_shape[-1], activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        attention_weights = self.attention(inputs)
        return inputs * attention_weights

# Feature creation function
def create_features(X_tilt, X_accel):
    if len(X_tilt.shape) == 1:
        X_tilt = X_tilt.reshape(1, -1)
    if len(X_accel.shape) == 1:
        X_accel = X_accel.reshape(1, -1)

    tilt_patterns = np.zeros((X_tilt.shape[0], 5))
    tilt_patterns[:, 0] = X_tilt[:, 0] * X_tilt[:, 1]
    tilt_patterns[:, 1] = X_tilt[:, 2] * X_tilt[:, 3]
    tilt_patterns[:, 2] = X_tilt[:, 4] * X_tilt[:, 5]
    tilt_patterns[:, 3] = X_tilt[:, 6] * X_tilt[:, 7]
    tilt_patterns[:, 4] = np.mean(X_tilt, axis=1)

    accel_magnitude = np.sqrt(np.sum(X_accel**2, axis=1, keepdims=True))
    accel_stats = np.concatenate([
        X_accel,
        accel_magnitude,
        np.abs(X_accel),
    ], axis=1)

    return tilt_patterns, accel_stats

# Register custom layers
get_custom_objects().update({
    'AttentionBlock': AttentionBlock,
    'SEBlock': SEBlock
})

# Load the model
loaded_model = tf.keras.models.load_model('model.h5')

# Flask app setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract tilt and accel data from JSON input
        tilt_input = np.array(data['tilt_input'])
        accel_input = np.array(data['accel_input'])

        # Create features
        tilt_patterns, accel_stats = create_features(tilt_input, accel_input)

        # Reshape tilt_input
        tilt_input = tilt_input.reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict([tilt_input, tilt_patterns, accel_stats])

        # Extract predicted class
        predicted_class_index = int(np.argmax(prediction))

        return jsonify({
            'predicted_class': predicted_class_index,
            'prediction': prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return 'Hello, this is our project. It is a Flask API for project, that predicts the class of a given input.'

@app.route('/about')
def about():
    return 'Author: Loc Phan Anh'


if __name__ == "__main__":
    app.run()
