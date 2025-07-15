# TFM_KS

## Clasificación multiclase (3 clases) de cáncer de próstata a partir de imágenes de resonancia magnética

### Estructura del repositorio

- **PC_predict.py**: Script a ejecutar para predecir las clases de cáncer junto con sus probabilidades estimadas a partir de imágenes de resonancia magnética.
- **Prostate_cancer_model.ipynb**: Script donde aparece el código principal utilizado para la creación del modelo predictivo
- **scaler_vgg16.joblib**: Escalador para normalizar los embeddings.
- **pca_vgg16_250c.joblib**: Objeto PCA entrenado para reducción de dimensionalidad.
- **modelo_svm_vgg16_pca250.joblib**: Modelo SVM entrenado para la predicción final.
- **images/**: Carpeta donde deben colocarse las imágenes a predecir (formato .jpg).
- **Prostate_Dataset/**: Carpeta donde se encuentran los datos de entrada del modelo divididos en train y test (formato .jpg).


### Instrucciones de ejecución

1. Coloca las imágenes (.jpg) a predecir en la carpeta /data/images/.
2. Abre la terminal (en mi caso he utilizado el entorno Anaconda).
3. Ejecuta el script con "python PC_predict.py"
4. El script imprimirá por pantalla la clase predicha y las probabilidades por clase para cada imagen.

### Requisitos principales

- Python 3.11.4 (recomendada, ya que es la utilizada en este caso)
- Anaconda (opcional, yo lo he utilizado)
- Librerías necesarias para la ejecución:
    - numpy
    - pandas
    - scikit-learn
    - tensorflow
    - joblib
