import sys
import os
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess_captions import load_captions, create_tokenizer

# === CONFIG ===
BASE_DIR = 'c:/image-captioning-project'
CAPTIONS_PATH = os.path.join(BASE_DIR, 'flickr8k', 'captions.txt')
FEATURES_PATH = os.path.join(BASE_DIR, 'flickr8k', 'features.pkl')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'backend', 'model', 'tokenizer.pkl')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'backend', 'model', 'image_caption_model.h5')
MAX_LENGTH_PATH = os.path.join(BASE_DIR, 'backend', 'model', 'max_length.pkl')
EPOCHS = 100
BATCH_SIZE = 64

# === Create necessary directories ===
os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)

# === Load features and captions ===
print("[INFO] Loading image features...")
with open(FEATURES_PATH, 'rb') as f:
    features = pickle.load(f)

print("[INFO] Loading captions...")
captions = load_captions(CAPTIONS_PATH)

# === Filter captions to only available features ===
captions = {k: captions[k] for k in captions if k in features}
print(f"[INFO] Filtered captions to {len(captions)} available images.")

# === Tokenize and save tokenizer ===
print("[INFO] Creating tokenizer...")
tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1

with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"[INFO] Vocabulary size: {vocab_size}")
print(f"[INFO] Tokenizer saved to {TOKENIZER_PATH}")

# === Calculate max caption length and save ===
all_captions = []
for key in captions:
    all_captions.extend(captions[key])
max_length = max(len(caption.split()) for caption in all_captions)

with open(MAX_LENGTH_PATH, 'wb') as f:
    pickle.dump(max_length, f)

print(f"[INFO] Max sequence length: {max_length}")
print(f"[INFO] Max sequence length saved to {MAX_LENGTH_PATH}")

# === Data generator ===
def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size):
    while True:
        keys = list(descriptions.keys())
        random.shuffle(keys)  # Shuffle the keys every epoch
        X1, X2, y = [], [], []
        for key in keys:
            if key not in features:
                continue
            for desc in descriptions[key]:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key])
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield ([np.array(X1), np.array(X2)], np.array(y))
                        X1, X2, y = [], [], []

# === Define model ===
def define_model(vocab_size, max_length):
    # Feature extractor (image)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence processor (text)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (combine)
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001))
    return model

# === Build and train model ===
print("[INFO] Building model...")
model = define_model(vocab_size, max_length)
model.summary()

steps = sum(len(c) for c in captions.values()) // BATCH_SIZE
generator = data_generator(captions, features, tokenizer, max_length, vocab_size, BATCH_SIZE)

# Callbacks
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=7, restore_best_weights=True, verbose=1)

print("[INFO] Starting training...")
model.fit(generator, epochs=EPOCHS, steps_per_epoch=steps, callbacks=[checkpoint, early_stopping], verbose=1)

print(f"[INFO] Training complete. Best model saved at {MODEL_SAVE_PATH}")