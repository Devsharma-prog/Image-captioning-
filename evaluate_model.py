import pickle
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Paths ===
MODEL_PATH = 'backend/model/image_caption_model.h5'
TOKENIZER_PATH = 'backend/model/tokenizer.pkl'
FEATURES_PATH = 'flickr8k/features.pkl'

# === Load the trained model and tokenizer ===
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

print("[INFO] Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

print("[INFO] Loading image features...")
with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)

# === BLEU Scoring Function ===
def calculate_bleu(reference_captions, generated_caption):
    references = [ref.split() for ref in reference_captions]
    candidate = generated_caption.split()
    smoothing = SmoothingFunction().method4

    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    return bleu1, bleu2, bleu3, bleu4

# === Caption Generation Function ===
def generate_caption(model, tokenizer, photo, max_length):
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
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# === Evaluate the Model ===
def evaluate_model(test_images, test_references, max_length):
    bleu_scores = []
    for image_id in test_images:
        # Generate caption for the image
        photo_feature = features[image_id].reshape((1, 2048))
        generated_caption = generate_caption(model, tokenizer, photo_feature, max_length)

        # Get reference captions
        reference_captions = test_references[image_id]

        # Calculate BLEU scores
        bleu1, bleu2, bleu3, bleu4 = calculate_bleu(reference_captions, generated_caption)
        bleu_scores.append((bleu1, bleu2, bleu3, bleu4))

        print(f"Image: {image_id}")
        print(f"Generated Caption: {generated_caption}")
        print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}\n")

    # Calculate average BLEU scores
    avg_bleu = np.mean(bleu_scores, axis=0)
    print(f"Average BLEU-1: {avg_bleu[0]:.4f}, BLEU-2: {avg_bleu[1]:.4f}, BLEU-3: {avg_bleu[2]:.4f}, BLEU-4: {avg_bleu[3]:.4f}")

# === Main ===
if __name__ == "__main__":
    # Test dataset
    test_images = ['1000268201_693b08cb0e.jpg', '1001773457_577c3a7d70.jpg']  # Replace with your test image IDs
    test_references = {
        '1000268201_693b08cb0e.jpg': [
            "a child in a pink dress climbing stairs",
            "a girl in a pink dress climbing stairs"
        ],
        '1001773457_577c3a7d70.jpg': [
            "two dogs playing on the road",
            "a black dog and a white dog playing"
        ]
    }

    # Max sequence length (use the same value as during training)
    max_length = 37

    # Evaluate the model
    evaluate_model(test_images, test_references, max_length)