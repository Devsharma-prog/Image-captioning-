import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# === Paths ===
BASE_DIR = 'c:/image-captioning-project'
MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'model', 'image_caption_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'backend', 'model', 'tokenizer.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'flickr8k', 'features.pkl')

# === Load the trained model and tokenizer ===
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

print("[INFO] Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 37  # use same max_length from training

# === Caption generation function ===
def generate_caption(model, tokenizer, photo, max_length, num_captions=3):
    captions = []
    for _ in range(num_captions):
        in_text = 'startseq'
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        captions.append(final_caption)
    return captions

# === Beam Search Version ===
def generate_caption_beam_search(model, tokenizer, photo, max_length, beam_width=3, num_captions=3):
    all_captions = []
    for _ in range(num_captions):
        sequences = [[['startseq'], 0.0]]
        for _ in range(max_length):
            all_candidates = []
            for seq, score in sequences:
                sequence = tokenizer.texts_to_sequences([' '.join(seq)])[0]
                sequence = pad_sequences([sequence], maxlen=max_length)
                yhat = model.predict([photo, sequence], verbose=0)
                top_indices = np.argsort(yhat[0])[-beam_width:]
                for idx in top_indices:
                    word = tokenizer.index_word.get(idx)
                    if word is None:
                        continue
                    candidate = [seq + [word], score - np.log(yhat[0][idx])]
                    all_candidates.append(candidate)
            sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]
        final_caption = ' '.join(sequences[0][0])
        final_caption = final_caption.replace('startseq', '').replace('endseq', '').strip()
        all_captions.append(final_caption)
    return all_captions

# === Feature extraction for new images ===
def extract_features(image_path):
    model_cnn = InceptionV3(weights='imagenet')
    model_cnn = Model(inputs=model_cnn.input, outputs=model_cnn.layers[-2].output)  # Get bottleneck features
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model_cnn.predict(image, verbose=0)
    return feature

# === Test with an existing image feature ===
def test_existing_image(image_id):
    print(f"[INFO] Generating caption for existing image ID: {image_id}...")
    with open(FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)
    photo_feature = features[image_id].reshape((1, 2048))
    
    # Generate multiple captions (greedy search)
    captions = generate_caption(model, tokenizer, photo_feature, max_length, num_captions=3)
    for idx, caption in enumerate(captions):
        print(f"Generated Caption {idx + 1} (Greedy Search): {caption}")
    
    # Generate multiple captions (beam search)
    captions_beam = generate_caption_beam_search(model, tokenizer, photo_feature, max_length, beam_width=3, num_captions=3)
    for idx, caption in enumerate(captions_beam):
        print(f"Generated Caption {idx + 1} (Beam Search): {caption}")

# === Test with a new uploaded image ===
def test_new_image(image_path):
    print(f"[INFO] Generating caption for new image: {image_path}...")
    photo_feature = extract_features(image_path).reshape((1, 2048))
    
    # Generate multiple captions (greedy search)
    captions = generate_caption(model, tokenizer, photo_feature, max_length, num_captions=3)
    for idx, caption in enumerate(captions):
        print(f"Generated Caption {idx + 1} (Greedy Search): {caption}")
    
    # Generate multiple captions (beam search)
    captions_beam = generate_caption_beam_search(model, tokenizer, photo_feature, max_length, beam_width=3, num_captions=3)
    for idx, caption in enumerate(captions_beam):
        print(f"Generated Caption {idx + 1} (Beam Search): {caption}")

# === Main ===
if __name__ == "__main__":
    # Test with an existing image feature from dataset
    test_existing_image('667626_18933d713e.jpg')  # <-- Use a valid image ID from your dataset

    # Test with a completely new image
    test_new_image('C:/Screenshot 2025-02-20 170021.png')  # <-- Replace with your image path
