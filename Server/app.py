# app.py
import base64
import json
import io
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from kafka import KafkaProducer, KafkaConsumer
from PIL import Image
import numpy as np
import cv2
print("importing tensorflow...")
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda m: json.dumps(m).encode('utf-8')
)

# Global variable to store the latest detection results
latest_results = {"count": 0, "timestamp": None}

# Load a pre-trained model for person detection
# We'll use SSD MobileNet from TensorFlow
print("Loading TensorFlow model...")
model = tf.saved_model.load('./ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/')
detect_fn = model.signatures['serving_default']

def preprocess_image(image_data):
    """Preprocess the image data for the model."""
    # Remove the "data:image/jpeg;base64," prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64 to binary
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # The model expects RGB, ensure we have the right format
    if image_np.shape[-1] == 4:  # If RGBA, convert to RGB
        image_np = image_np[:, :, :3]
    
    # Expand dimensions for batch processing
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    return input_tensor

def count_people(image_data):
    """Count the number of people in an image using the pre-trained model."""
    input_tensor = preprocess_image(image_data)
    
    # Perform detection
    detections = detect_fn(input_tensor)
    
    # Extract detection results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
                  for key, value in detections.items()}
    
    # Filter for person class (class 1 in COCO dataset)
    person_indices = np.where((detections['detection_classes'] == 1) & 
                             (detections['detection_scores'] >= 0.5))[0]
    
    return len(person_indices)

# Kafka consumer thread to process frames
def kafka_consumer_thread():
    consumer = KafkaConsumer(
        'video-frames',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    
    for message in consumer:
        try:
            # Process the frame
            frame_data = message.value['frame']
            client_id = message.value['client_id']
            
            # Count people in the frame
            people_count = count_people(frame_data)
            
            # Update latest results
            global latest_results
            latest_results = {
                "count": people_count,
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id
            }
            
            # Send results to Kafka
            producer.send('detection-results', {
                "count": people_count,
                "timestamp": latest_results["timestamp"],
                "client_id": client_id
            })
            
            print(f"Detected {people_count} people for client {client_id}")
            
        except Exception as e:
            print(f"Error processing frame: {e}")

# Start Kafka consumer thread
consumer_thread = threading.Thread(target=kafka_consumer_thread)
consumer_thread.daemon = True
consumer_thread.start()

@app.route('/send-frame', methods=['POST'])
def send_frame():
    try:
        data = request.json
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({"success": False, "error": "No frame data provided"}), 400
        
        # Generate a client ID or use one from the request
        client_id = data.get('client_id', request.remote_addr)
        
        # Send frame to Kafka
        producer.send('video-frames', {
            "frame": frame_data,
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id
        })
        
        return jsonify({"success": True})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get-results', methods=['GET'])
def get_results():
    # Return the latest detection results
    return jsonify(latest_results)

if __name__ == '__main__':
    # Download the model if needed
    # (This would typically be done separately)
    
    print("Starting server on port 5001...")
    app.run(host='0.0.0.0', port=5001, ssl_context=('./Client/localhost.pem', './Client/localhost-key.pem'), debug=True)