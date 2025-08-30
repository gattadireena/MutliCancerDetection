# 🧬 Multicancer Detection Web App (Brain, Lung, Colon, Kidney)

## 📖 Overview
This project is a **Multicancer Detection System** that uses **deep learning (CNN, ResNet50)** to detect:
- 🧠 Brain Tumor  
- 🫁 Lung Cancer  
- 🌿 Colon Cancer  
- 🩺 Kidney Cancer  

Model training was done in **Google Colab** (for GPU support), and the trained models were later integrated into a **Flask-based web app** (run locally in VS Code).  
Users can upload an image, select the cancer type, and view real-time predictions.

---

## 🚀 Features
- Train cancer detection models in Google Colab notebooks  
- Export trained models (`.h5`) and integrate into Flask app  
- Upload image + select model type (Brain, Lung, Colon, Kidney)  
- Real-time predictions with preview in browser  
- Achieved **high accuracy**:  
  - Colon → **99.8%**  
  - Brain → **98.3%**  
  - Lung → **98%**  
  - Kidney → **98%**  

---

## 📂 Project Structure
├── colab_notebooks/ # Google Colab codes for model training
│ ├── brain_cancer.ipynb
│ ├── lung_cancer.ipynb
│ ├── colon_cancer.ipynb
│ └── kidney_cancer.ipynb
├── models/ # Exported trained models (.h5)
│ ├── brain_tumor_model.h5
│ ├── lung_cancer_model.h5
│ ├── colon_cancer_model.h5
│ └── kidney_cancer_model.h5
├── app.py # Flask backend
├── index.html # Frontend (upload + preview + results)
├── static/ # (Optional) CSS/JS/image assets
├── templates/ # (Optional) Flask templates
└── README.md # Documentation


---

## ⚙️ Tech Stack
- **Model Training (Colab):** Python, TensorFlow, Keras, OpenCV  
- **Web Application (VS Code):** Flask, HTML, CSS, JavaScript  
- **Models:**  
  - Brain → Custom CNN  
  - Lung → ResNet50  
  - Colon → ResNet50  
  - Kidney → ResNet50  

---

## ▶️ How to Use

### 🔹 1. Train Models (Google Colab)
- Open the provided notebooks in `colab_notebooks/`  
- Train each model (Brain, Lung, Colon, Kidney)  
- Save/export trained models as `.h5` files  
- Download them and move to the `models/` folder  

### 🔹 2. Run Flask App (VS Code / Local)
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multicancer-detection-webapp.git
   cd multicancer-detection-webapp
2. Install dependencies:
   pip install -r requirements.txt

3. Ensure models are in models/ folder:
   models/
├── brain_tumor_model.h5
├── lung_cancer_model.h5
├── colon_cancer_model.h5
└── kidney_cancer_model.h5

4. Run Flask app:
   python app.py

5. Open browser at:
   http://127.0.0.1:5000

6. Upload an image → Select cancer type → Get prediction ✅

📊 Results
Colon Cancer → 99.8% Accuracy
Brain Tumor → 98.3% Accuracy
Lung Cancer → 98% Accuracy
Kidney Cancer → 98% Accuracy


📌 Future Work
Deploy app on AWS / GCP / Heroku
Blockchain integration for secure medical record storage
UI upgrade with React.js / Streamlit
Unified ensemble model for multi-cancer detection


🏆 Acknowledgements
Datasets: Kaggle medical datasets (Brain, Lung, Colon, Kidney)
Tools: Google Colab (GPU), VS Code (Flask)
Libraries: TensorFlow, Keras, Flask, OpenCV
