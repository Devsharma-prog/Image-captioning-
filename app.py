from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
CORS(app)

# Model and tokenizer paths
model_path = os.environ.get('MODEL_PATH', 'C:/image-captioning-project/backend/model/image_caption_model.h5')
tokenizer_path = os.environ.get('TOKENIZER_PATH', 'C:/image-captioning-project/backend/model/tokenizer.pkl')

# Load model and tokenizer
try:
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Model and tokenizer loaded successfully from {model_path} and {tokenizer_path}")
except Exception as e:
    print(f"Error loading model or tokenizer: {str(e)}")
    model = None
    tokenizer = None

# Load feature extractor
try:
    feature_extractor = InceptionV3(weights='imagenet')
    feature_extractor = Model(inputs=feature_extractor.input, outputs=feature_extractor.layers[-2].output)
    print("Feature extractor loaded successfully")
except Exception as e:
    print(f"Error loading feature extractor: {str(e)}")
    feature_extractor = None

max_length = 37  # Make sure this matches your training config

def preprocess_image(image_stream):
    img = Image.open(image_stream)
    img = img.convert('RGB')
    img = img.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_features(image_stream):
    if feature_extractor is None:
        raise ValueError("Feature extractor not loaded")
    img = preprocess_image(image_stream)
    feature = feature_extractor.predict(img, verbose=0)
    return feature

def generate_caption(photo):
    if model is None or tokenizer is None:
        raise ValueError("Model or tokenizer not loaded")

    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return [final_caption]

@app.route('/generate-caption/', methods=['POST'])
def generate_caption_from_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        print('Received image file:', file.filename)
        features = extract_features(file.stream)

        captions = generate_caption(features)
        print("Generated captions:", captions)

        return jsonify({'captions': captions})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
