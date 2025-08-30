from flask import Flask, request, render_template, jsonify, g
import tensorflow as tf
import numpy as np
import cv2
import os
import pickle
import logging
import json
from werkzeug.utils import secure_filename
import uuid
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to models and preprocessing files
brain_tumor_model_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\Brain_tumor_model.h5"
kidney_cancer_model_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\kidneyy_cancer_model.h5"
kidney_cancer_preprocessor_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\kidcancerimage_preprocessor.pkl"
class_labels_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\class_labels.json"
colon_cancer_model_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\colon_cancer_classifier.h5"
colon_cancer_preprocessor_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\COLON_preprocess_config.pkl"
lung_cancer_model_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\lung1_cancer_model.h5"
lung_class_indices_path = r"C:\Users\SAI RAM\Downloads\BRAIN TUMORRR\lungclass_indices.json"

# Labels
brain_tumor_labels = ["Affected", "Healthy"]
kidney_cancer_labels = {0: "Normal", 1: "Tumor"}
colon_cancer_labels = ["Colon_Adenocarcinoma", "Colon_Benign_Tissue"]
lung_cancer_labels = ["Lung-Benign_Tissue", "Lung_Adenocarcinoma", "Lung_Squamous_Cell_Carcinoma"]

# Load models and preprocessors
def load_model():
    if not hasattr(g, 'brain_tumor_model'):
        g.brain_tumor_model = tf.keras.models.load_model(brain_tumor_model_path)
        g.kidney_cancer_model = tf.keras.models.load_model(kidney_cancer_model_path)
        g.colon_cancer_model = tf.keras.models.load_model(colon_cancer_model_path)
        g.lung_cancer_model = tf.keras.models.load_model(lung_cancer_model_path)

        with open(kidney_cancer_preprocessor_path, 'rb') as file:
            g.kidney_cancer_preprocessor = pickle.load(file)

        with open(class_labels_path, 'r') as file:
            g.kidney_cancer_class_labels = json.load(file)

        with open(colon_cancer_preprocessor_path, 'rb') as file:
            g.colon_cancer_preprocessor = pickle.load(file)

        with open(lung_class_indices_path, 'r') as file:
            g.lung_cancer_class_labels = json.load(file)

    return (
        g.brain_tumor_model,
        g.kidney_cancer_model,
        g.kidney_cancer_preprocessor,
        g.kidney_cancer_class_labels,
        g.colon_cancer_model,
        g.colon_cancer_preprocessor,
        g.lung_cancer_model,
        g.lung_cancer_class_labels
    )

# Image preprocessing
def preprocess_image(image_path, model_type="brain_tumor"):
    img = cv2.imread(image_path)

    if model_type == "brain_tumor":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=[0, -1])

    else:  # kidney_cancer, colon_cancer, lung_cancer
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

    return img

# Helpers
def is_image(file):
    return file and file.filename.lower().endswith(('png', 'jpg', 'jpeg'))

def save_temp_file(file):
    temp_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    temp_filepath = os.path.join("static/uploads", temp_filename)
    file.save(temp_filepath)
    return temp_filepath

@app.route("/", methods=["GET", "POST"])
def index():
    logging.info("Received a request")

    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"})

        if not is_image(file):
            return jsonify({"error": "Invalid file type. Please upload an image."})

        filepath = save_temp_file(file)

        try:
            model_type = request.form.get("model_type", "brain_tumor")
            img = preprocess_image(filepath, model_type)

            (
                brain_tumor_model,
                kidney_cancer_model,
                kidney_cancer_preprocessor,
                kidney_cancer_class_labels,
                colon_cancer_model,
                colon_cancer_preprocessor,
                lung_cancer_model,
                lung_cancer_class_labels
            ) = load_model()

            if model_type == "brain_tumor":
                prediction = brain_tumor_model.predict(img)
                label = brain_tumor_labels[prediction.argmax()]

            elif model_type == "kidney_cancer":
                prediction = kidney_cancer_model.predict(img)
                predicted_class = 1 if prediction[0][0] > 0.5 else 0
                label = kidney_cancer_labels[predicted_class]

            elif model_type == "colon_cancer":
                prediction = colon_cancer_model.predict(img)
                label = colon_cancer_labels[prediction.argmax()]

            elif model_type == "lung_cancer":
                prediction = lung_cancer_model.predict(img)
                predicted_index = prediction.argmax()
                label = lung_cancer_labels[predicted_index]

            else:
                return jsonify({"error": f"Invalid model_type: {model_type}"})

            return jsonify({"prediction": label, "image_url": filepath})

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": f"An error occurred while processing the image: {str(e)}"})

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
