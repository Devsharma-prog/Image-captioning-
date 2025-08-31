import os
import sys
import pickle
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# === Load feature extractor model ===
def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

model = load_feature_extractor()

def extract_features(directory, output_path, batch_size=100):
    features = {}
    skipped_files = []

    all_files = os.listdir(directory)
    total_images = 0
    print(f"🔍 Total files in directory: {len(all_files)}")

    for img_name in all_files:
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_images += 1
            img_path = os.path.join(directory, img_name)
            try:
                img = image.load_img(img_path, target_size=(299, 299))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                feature = model.predict(img_array, verbose=0)
                features[img_name] = feature.flatten()  # Or just feature[0] if you want (2048,) shape directly

                if total_images % batch_size == 0:
                    with open(output_path, "wb") as f:
                        pickle.dump(features, f)
                    print(f"✅ Saved intermediate features for {len(features)} images.")

            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")
                skipped_files.append((img_name, str(e)))
        else:
            print(f"⚠️ Skipping non-image file: {img_name}")
            skipped_files.append((img_name, "Not an image file"))

    # Final save
    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"✅ Completed feature extraction.")
    print(f"✅ Extracted features for {len(features)} images out of {total_images} processed files.")
    
    if skipped_files:
        print(f"⚠️ Skipped {len(skipped_files)} files:")
        for file, reason in skipped_files:
            print(f"   - {file}: {reason}")

    return features

if __name__ == "__main__":
    dataset_path = r'c:\image-captioning-project\flickr8k\Images'

    output_path = r'c:\image-captioning-project\flickr8k\features.pkl' 

    print(f"🔍 Checking dataset path: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"❌ The dataset path '{dataset_path}' does not exist.")
        sys.exit()

    if not os.listdir(dataset_path):
        print(f"❌ The dataset path '{dataset_path}' is empty.")
        sys.exit()

    print(f"🔍 Extracting features from images...")
    features = extract_features(dataset_path, output_path)

    print(f"✅ All done! Extracted features saved to: {output_path}")