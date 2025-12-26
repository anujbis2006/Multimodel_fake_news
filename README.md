# Multimodel_fake_news
# 📰 Multimodal Fake News Detection (Text + Image)

A deep learning–based **multimodal fake news detection system** that analyzes **both textual content and images** to classify news as **Real** or **Fake**.

This project uses a **late-fusion architecture** combining:
- 📄 Text features (LSTM)
- 🖼 Image features (CNN)
- 🔗 Fusion layers for final decision

---

## 🚀 Demo (Streamlit App)

The project includes an interactive **Streamlit web app** where users can:
- Enter news text / title
- Upload an image
- Get a **Real/Fake prediction with confidence score**

> Example output:
> - ✅ REAL NEWS — Confidence: 50.65%

---

## 🧠 Model Architecture

### 🔹 Text Branch
- Tokenization + Padding
- Embedding Layer
- LSTM (frozen during multimodal training)

### 🔹 Image Branch
- CNN with multiple Conv + Pool layers
- Feature compression layer
- Frozen convolutional backbone

### 🔹 Multimodal Fusion
- Concatenation of text + image embeddings
- Fully connected fusion layers
- Sigmoid output (binary classification)

---

## 📂 Project Structure

```text
Multimodal_fake_news/
│  
├── app.py                     # Streamlit app
├── scripts/
│   ├── train_text_lstm.py
│   ├── train_image_cnn.py
│   ├── train_multimodal_model.py
│   └── evaluate_models.py
│
├── outputs/
│   └── tokenizer.pkl          # Saved tokenizer
│
├── models/
│   ├── lstm_model.keras
│   ├── cnn_model.keras
│   └── multimodal_model.keras
│
├── data/                      # (ignored in GitHub)
│   └── fakeddit_subset/
│
├── .gitignore
└── README.md

📊 Dataset

Fakeddit (Subset)

Contains:

News titles/text

Associated images

Binary labels (Real / Fake)

📌 Note:
Due to size constraints, the dataset and trained models are not included in this repository.
You can download the dataset separately and place it inside the data/ directory.

🏋️ Training Strategy

Text-only model trained separately

Image-only model trained separately

Multimodal model:

Base branches frozen

Only fusion layers trained

Prevents overfitting and label leakage

📈 Evaluation Results (Sample)
Model Type	Accuracy
Text-only	~55%
Image-only	~44%
Multimodal	~56%

Multimodal learning improves robustness by combining visual and textual cues.

🧪 Example Input Format

Text

"myanmar court sentences two reuters journalists"


Image

Uploaded image related to the news headline


Model Output

REAL NEWS — Confidence: 50.65%

