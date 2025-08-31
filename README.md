# 🧠 Image Captioning Project

A full-stack image captioning app that generates captions for uploaded images using a deep learning model (CNN + LSTM).

## 🧰 Tech Stack

- Python (TensorFlow, Keras, Flask)
- HTML, CSS, JS (Frontend)
- Flickr8k Dataset

## 📁 Folder Structure
image-captioning-project/ ├── backend/ │ ├── model/ # Model architecture and weights │ ├── utils/ # Preprocessing utilities │ ├── app.py # Flask backend server │ └── README.md # Project details ├── flickr8k/ # Dataset folder (images + captions) │ ├── Images/ │ ├── captions.txt │ └── Flickr8k.trainImages.txt (etc.) ├── frontend/ # Frontend UI │ ├── index.html │ ├── style.css │ ├── script.js ├── requirements.txt # Python libraries └── README.md # (You're reading it!)

yaml
Copy code

---

### 💻 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the backend server
cd backend
python app.py

# 3. Open the frontend
# Just open index.html in any browser (e.g. Chrome)
📸 Model Info
CNN: Pre-trained InceptionV3 extracts image features.

RNN: LSTM generates natural language captions.

Dataset: Flickr8k — 8,000 images with 5 captions each.

🙋‍♂️ Contributors
[Your Name]

[Teammates' Names]

📄 License
This project is for educational purposes only.
