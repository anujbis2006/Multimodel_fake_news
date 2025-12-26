# Multimodal Fake News Detection

A deep learning project that combines **text** and **image analysis** to detect fake news using a late-fusion multimodal architecture. This system leverages both textual content and visual information to improve fake news detection accuracy.

## 🎯 Project Overview

This project implements a multimodal neural network that processes both text and images simultaneously to classify news content as authentic or fake. The model uses:
- **Text Branch**: LSTM (Long Short-Term Memory) network for sequential text analysis
- **Image Branch**: CNN (Convolutional Neural Network) for visual feature extraction
- **Fusion Layer**: Late-fusion architecture combining both modalities

## ✨ Key Features

- **Multimodal Architecture**: Combines text and image information for comprehensive fake news detection
- **Late Fusion Design**: Independently processes text and images before combining them
- **Frozen Base Layers**: Pre-trained embedding and CNN backbone with trainable fusion layers
- **Comprehensive Data Handling**: Supports JSONL format with image references
- **Model Evaluation**: Complete evaluation metrics including accuracy, precision, recall, and F1-score
- **Checkpoint Saving**: Automatic model checkpointing based on validation accuracy

## 📊 Model Performance

**Validation Results:**
- **Best Validation Accuracy**: 57.67%
- **Final Training Accuracy**: 61.70%
- **Precision**: 0.4844
- **Recall**: 0.4170
- **F1-Score**: 0.4482

**Model Statistics:**
- Total Parameters: 4,044,673 (15.43 MB)
- Trainable Parameters: 139,969 (546.75 KB)
- Non-trainable Parameters: 3,904,704 (14.90 MB)

## 🏗️ Architecture

```
Text Branch:                          Image Branch:
Input (128,)                         Input (128, 128, 3)
    ↓                                     ↓
Embedding (128×128)                  Conv2D (32 filters)
    ↓                                     ↓
LSTM (128 units) [FROZEN]            MaxPooling → Conv2D → MaxPooling
    ↓                                     ↓
Text Expand (256 dim)                Image Dense (128)
    ↓                                     ↓
         Dense (64 dim) [COMPRESS IMAGE]
              ↓
         Concatenate (320 dim)
              ↓
         Fusion Dense Layers (256 → 64)
              ↓
         Output (Sigmoid) → Binary Classification
```

## 📦 Project Structure

```
Multimodal-fake-news/
├── app.py                              # Main application file
├── README.md                           # Project documentation
├── data/
│   └── fakeddit_subset/
│       ├── training_data_fakeddit.jsonl
│       ├── validation_data_fakeddit.jsonl
│       ├── image_folder/               # Training images
│       └── validation_image/           # Validation images
├── models/
│   ├── cnn_model.keras
│   ├── lstm_model.keras
│   └── multimodal_model.keras         # Best trained model
├── outputs/
│   ├── tokenizer.pkl                 # Text tokenizer
│   ├── image_ids.csv
│   ├── labels.npy
│   └── text_sequence.npy
└── scripts/
    ├── train_multimodal_model.py     # Main training script
    ├── train_image_cnn.py
    ├── train_text_lstm.py
    ├── evaluate_all_models.py
    ├── evaluate_text_model.py
    ├── text_preprocessing.py
    ├── text_eda.py
    └── debug_inspect_text.py
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- TensorFlow 2.x
- scikit-learn
- numpy

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/Multimodal-fake-news.git
cd Multimodal-fake-news
```

2. **Create a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📚 Usage

### Training the Multimodal Model

```bash
python scripts/train_multimodal_model.py \
  --train data/fakeddit_subset/training_data_fakeddit.jsonl \
  --val data/fakeddit_subset/validation_data_fakeddit.jsonl \
  --img-dir data/fakeddit_subset/image_folder \
  --val-img-dir data/fakeddit_subset/validation_image \
  --tokenizer outputs/tokenizer.pkl \
  --pad 128 \
  --epochs 12 \
  --exclude-model-answers
```

### Command Line Arguments

- `--train`: Path to training JSONL file (required)
- `--val`: Path to validation JSONL file (required)
- `--img-dir`: Path to training images directory
- `--val-img-dir`: Path to validation images directory
- `--tokenizer`: Path to pickled tokenizer file
- `--pad`: Sequence padding length (default: 128)
- `--epochs`: Number of training epochs (default: 12)
- `--batch`: Batch size (default: 32)
- `--embed`: Embedding dimension (default: 128)
- `--lstm`: LSTM units (default: 128)
- `--image-size`: Image size (default: 128 128)
- `--image-compress-dim`: Image feature compression dimension (default: 64)
- `--text-expand-dim`: Text feature expansion dimension (default: 256)
- `--fusion-dim1`: First fusion layer size (default: 256)
- `--fusion-dim2`: Second fusion layer size (default: 64)
- `--exclude-model-answers`: Flag to exclude model answers from text extraction

### Training the Text LSTM Model

```bash
python scripts/train_text_lstm.py \
  --train data/fakeddit_subset/training_data_fakeddit.jsonl \
  --val data/fakeddit_subset/validation_data_fakeddit.jsonl \
  --tokenizer outputs/tokenizer.pkl
```

### Training the Image CNN Model

```bash
python scripts/train_image_cnn.py \
  --train-dir data/fakeddit_subset/image_folder \
  --val-dir data/fakeddit_subset/validation_image
```

### Evaluation

```bash
python scripts/evaluate_all_models.py
```

## 📄 Data Format

The JSONL files should contain records with the following structure:
- Text fields: title, caption, content, messages, etc.
- Image ID: References to image files
- Labels: fake/real classification or model answers

Example record:
```json
{
  "title": "Breaking news...",
  "content": "Article content...",
  "image_id": "image_filename.jpg",
  "label": 1
}
```

## 🔧 Technical Details

### Text Processing
- Tokenization with custom SimpleTokenizer
- Sequence padding to fixed length (default: 128)
- Embedding layer with configurable dimensions

### Image Processing
- JPEG decoding with TensorFlow
- Resizing to specified dimensions (default: 128×128)
- Normalization to [0, 1] range

### Training Strategy
- **Frozen Base Layers**: Embedding, LSTM, and CNN backbone layers are non-trainable
- **Trainable Fusion Layers**: Only the fusion and output layers are trained
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## 📈 Results Analysis

The model shows:
- Gradual improvement during training
- Best validation accuracy at epoch 11 (57.67%)
- Moderate precision-recall balance
- Room for improvement through:
  - Data augmentation
  - Hyperparameter tuning
  - Unfreezing base layers after initial training
  - Class balancing techniques

## 🛠️ Dependencies

See `requirements.txt` for complete list. Main dependencies:
- TensorFlow >= 2.10
- NumPy
- scikit-learn
- Pillow

## 📝 File Descriptions

| File | Purpose |
|------|---------|
| `train_multimodal_model.py` | Main script for training late-fusion multimodal model |
| `train_text_lstm.py` | LSTM model training for text classification |
| `train_image_cnn.py` | CNN model training for image classification |
| `evaluate_all_models.py` | Comprehensive evaluation of all trained models |
| `text_preprocessing.py` | Text cleaning and preprocessing utilities |
| `text_eda.py` | Exploratory data analysis for text |
| `debug_inspect_text.py` | Debugging utility for text inspection |

## 🚦 Future Improvements

- [ ] Implement attention mechanisms
- [ ] Add data augmentation techniques
- [ ] Experiment with different architectures (e.g., Vision Transformers)
- [ ] Fine-tune pre-trained models (BERT, ResNet)
- [ ] Implement explainability features (Grad-CAM, attention visualization)
- [ ] Add web interface for predictions
- [ ] Deploy model as REST API

## 📜 License

This project is open source and available under the MIT License.

## 👤 Author

Created by Anuj Biswas

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📧 Contact

For questions or collaboration, feel free to reach out.

---

**Last Updated**: December 26, 2025
**Status**: Complete and Ready for Use
