import os
import re
import string
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Load and parse captions file ===
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            image_id, caption = tokens
            image_id = image_id.split('#')[0]
            if image_id not in captions:
                captions[image_id] = []
            cleaned_caption = 'startseq ' + clean_text(caption) + ' endseq'
            captions[image_id].append(cleaned_caption)
    print(f"[INFO] Loaded captions for {len(captions)} images.")
    return captions

# === Clean caption text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# === Create tokenizer ===
def create_tokenizer(captions_dict, vocab_size_limit=None):
    all_captions = []
    for key in captions_dict:
        all_captions.extend(captions_dict[key])
    tokenizer = Tokenizer(num_words=vocab_size_limit, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

# === Create padded sequences (for generator) ===
def data_generator(tokenizer, max_length, descriptions, photo_features, vocab_size, batch_size=64):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            photo = photo_features.get(key)
            if photo is None:
                print(f"⚠️ Warning: Missing features for image {key}")
                continue
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        yield np.array(X1), np.array(X2), np.array(y)
                        X1, X2, y = [], [], []
                        n = 0

# === Save tokenizer ===
def save_tokenizer(tokenizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)

# === Load tokenizer ===
def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Paths
    base_path = os.path.join("c:", "image-captioning-project", "flickr8k")
    captions_file = r'c:\image-captioning-project\flickr8k\captions.txt'
    features_file = r'c:\image-captioning-project\flickr8k\features.pkl'
    tokenizer_file = r'c:\image-captioning-project\flickr8k\tokenizer.pkl'

    # Step 1: Load captions
    print("🔍 Loading captions...")
    captions = load_captions(captions_file)

    # Step 2: Load image features
    print("🔍 Loading image features...")
    with open(features_file, "rb") as f:
        photo_features = pickle.load(f)

    # Step 3: Filter captions to match features
    print("🔍 Filtering captions to match available features...")
    filtered_captions = {k: captions[k] for k in captions if k in photo_features}

    # Step 4: Create tokenizer
    print("🔍 Creating tokenizer...")
    vocab_limit = 5000
    tokenizer = create_tokenizer(filtered_captions, vocab_size_limit=vocab_limit)
    save_tokenizer(tokenizer, tokenizer_file)
    print(f"✅ Tokenizer created with vocab size {len(tokenizer.word_index) + 1}. Saved to {tokenizer_file}")

    # Step 5: Determine max caption length
    all_captions = []
    for key in filtered_captions:
        all_captions.extend(filtered_captions[key])
    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"🔍 Max caption length: {max_length} words.")

    # Step 6: Test data generator
    print("🔍 Testing data generator...")
    vocab_size = len(tokenizer.word_index) + 1
    generator = data_generator(tokenizer, max_length, filtered_captions, photo_features, vocab_size)

    # Example: Process a few batches for testing
    for i, (X1_batch, X2_batch, y_batch) in enumerate(generator):
        print(f"✅ Processed batch {i+1}: X1={X1_batch.shape}, X2={X2_batch.shape}, y={y_batch.shape}")
        if i == 2:  # Just a few batches for testing
            break

    print("✅ Preprocessing complete.")