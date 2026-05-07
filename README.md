# 🧠 Multimodal Deepfake Detection System

## 📌 Overview
This project presents a **multimodal deepfake detection system** designed to identify AI-generated content across **text, image, audio, and video modalities**.

Unlike traditional single-modality approaches, this system integrates multiple specialized models and combines their outputs using an **accuracy-weighted fusion strategy**, improving reliability and robustness.

---

## 🎯 Objectives
- Develop independent models for text, image, audio, and video detection
- Evaluate each modality using classification metrics
- Implement an interpretable fusion strategy
- Analyze robustness under different experimental conditions

---

## 📊 Dataset Sources

- **Text Dataset**
  - https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text  

- **Image Dataset**
  - https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images  

- **Audio Dataset**
  - https://www.kaggle.com/datasets/adarshsingh0903/audio-deepfake-detection-dataset  

- **Video Dataset**
  - https://www.kaggle.com/datasets/reubensuju/celeb-df-v2  

 Note: Due to runtime and computational constraints, subsets of these datasets were used.

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Deep Learning:** TensorFlow, Keras, PyTorch  
- **NLP:** HuggingFace Transformers (BERT)  
- **Image Processing:** OpenCV  
- **Audio Processing:** Librosa (MFCC)  
- **Visualization:** Matplotlib, Seaborn  

---

## ⚙️ Methodology

### 🔹 Text Detection
- Model: **BERT (bert-base-uncased)**
- Input: Tokenized text (max length = 128)
- Output: Binary classification (REAL / FAKE)

### 🔹 Image Detection
- Model: **Custom CNN**
- Architecture:
  - Conv2D(32) → Conv2D(64) → Conv2D(128)
  - Flatten → Dense(128) → Dropout(0.3)
  - Sigmoid Output
- Images resized to **128 × 128**

### 🔹 Audio Detection
- Feature Extraction: **MFCC (20 coefficients)**
- Model: CNN-based classifier
- Preprocessing:
  - Padding / truncation
  - Noise injection for robustness

### 🔹 Video Detection
- Model: **EfficientNet-B0**
- Approach:
  - Extract frames from videos
  - Face-centered preprocessing
  - Aggregate predictions across frames

---

## 🔗 Fusion Strategy

Final prediction is computed using an **accuracy-weighted fusion rule**:

P(x) = Σ (wᵢ * Pᵢ(x))

Where:
- Pᵢ(x) = probability from modality i
- wᵢ = accuracy-based weight

---

## 📈 Results

| Modality | Accuracy |
|----------|---------|
| Text (BERT) | 99.06% |
| Image (CNN) | 93.81% |
| Audio (MFCC + CNN) | 91.82% |
| Video (EfficientNet-B0) | 80.00% |
| Naive Fusion | 92.85% |
| **Weighted Fusion** | **99.31%** |

---

## 🧪 Robustness Testing

Robustness testing was conducted to evaluate model stability under variations such as **random seed, hyperparameters, and noise injection**.

### 🔹 Text Model (Seed Variation)
- Random State = 42 → Accuracy: **99.06%**
- Random State = 21 → Accuracy: **99.37%**

✔ Insight: Model remains highly stable across different splits :contentReference[oaicite:1]{index=1}  

---

### 🔹 Image Model (Learning Rate Tuning)
- Learning Rate = 0.001 → Accuracy ≈ **99.36%**
- Learning Rate = 0.0001 → Accuracy ≈ **94%**

✔ Insight: Performance is stable, but learning rate affects convergence and optimization :contentReference[oaicite:2]{index=2}  

---

### 🔹 Audio Model (Noise Injection)
- Noise Level = 0.005 → Accuracy ≈ **93%**
- Noise Level = 0.01 → Accuracy ≈ **74%**
- Additional runs → Accuracy ≈ **92%**

✔ Insight: Audio model is sensitive to noise due to smaller dataset size :contentReference[oaicite:3]{index=3}  

---

📌 **Conclusion:**  
High accuracy alone is not sufficient — robustness testing reveals model sensitivity and generalization behavior.

---

## ⚠️ Limitations
- Limited dataset size for audio and video modalities
- Fusion assumes independence between modalities
- Performance may vary across unseen datasets

---

## 🚀 Future Work
- Use larger datasets (Celeb-DF, DFDC full dataset)
- Add explainability (SHAP, Grad-CAM)
- Deploy as real-time detection system

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run notebooks
jupyter notebook

---

## 📂 Project Structure

```
Multimodal_Deepfake_Detection_System/
│
├── notebooks/
│   ├── fake_real_Text_Image_Audio.ipynb
│   └── fake_video_detection_fusion_model.ipynb
│
├── docs/
│   ├── Multimodal_Deepfake_Technical_Report.docx
│   ├── Tuning_Hyperparameters.docx
│   └── best_project_group-04-AI_certificate.pdf
│
├── results/
│   ├── ConfusionMatrix_Text.png
│   ├── ConfusionMatrix_Image.png
│   ├── ConfusionMatrix_Audio.png
│   ├── ConfusionMatrix_Video.png
│
├── README.md
├── requirements.txt
├── .gitignore
└── dataset links.md
```
---

## ⚙️ Hyperparameter Tuning
Detailed tuning experiments and configurations are documented in:

📄 docs/Tuning_Hyperparameters.docx

---

## 🏆 Key Takeaway
Multimodal learning significantly enhances deepfake detection performance. The use of **accuracy-weighted fusion** provides a simple yet effective way to combine predictions from multiple models.

---

## 📜 Recognition
- Recognized for academic project contribution
- Demonstrates real-world multimodal AI system development
