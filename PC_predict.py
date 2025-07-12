import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from PIL import Image
import pandas as pd

# ---- CONFIGURACIÓN ----
images_path = './data/images'
scaler_path = "scaler_vgg16.joblib"
pca_path = "pca_vgg16_250c.joblib"
svm_path = "modelo_svm_vgg16_pca250.joblib"
class_names = ["Healthy", "Not Significant", "Significant"]

# Buscar imágenes JPG en el directorio
image_files = [f for f in os.listdir(images_path) if f.lower().endswith('.jpg')]

if not image_files:
    print(f"No hay imágenes en el directorio indicado ({images_path})")
else:
    # Cargar transformadores y modelo
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    svm = joblib.load(svm_path)

    # Preparar modelo VGG16
    base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    embedding_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    # Procesar todas las imágenes
    probs_list = []
    pred_indices = []
    filenames = []

    for filename in image_files:
        img_path = os.path.join(images_path, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        embedding = embedding_model.predict(img_array, verbose=0)
        embedding_scaled = scaler.transform(embedding)
        embedding_pca = pca.transform(embedding_scaled)

        probs = svm.predict_proba(embedding_pca)[0]
        pred_class_idx = np.argmax(probs)

        probs_list.append(probs)
        pred_indices.append(pred_class_idx)
        filenames.append(filename)

    # Crear DataFrame como tu visualización
    df_pred = pd.DataFrame({
        "Image": filenames,
        "Predicted_Class": [class_names[i] for i in pred_indices],
        "Prob_Healthy": [probs[0] for probs in probs_list],
        "Prob_NotSignif": [probs[1] for probs in probs_list],
        "Prob_Significant": [probs[2] for probs in probs_list],
    })
    
    # Redondear y formatear probabilidades a 4 decimales (sin notación científica)
    cols_probs = ["Prob_Healthy", "Prob_NotSignif", "Prob_Significant"]
    for col in cols_probs:
        df_pred[col] = df_pred[col].map(lambda x: f"{x:.4f}")

    print("\n--- Probabilidades por imagen procesada ---")
    print(df_pred[["Image", "Predicted_Class", "Prob_Healthy", "Prob_NotSignif", "Prob_Significant"]].to_string(index=False))
