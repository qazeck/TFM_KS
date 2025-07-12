# TFM_KS

## Clasificación multiclase (3 clases) de cáncer de próstata a partir de imágenes de resonancia magnética

### Estructura del repositorio

- **PC_predict.py**: Script a ejecutar para predecir las clases de cáncer junto con sus probabilidades estimadas a partir de imágenes de resonancia magnética.
- **PC_model.ipynb**: Script donde aparece el código principal utilizado para la creación del modelo predictivo
- **Prostate_cancer_model.ipynb**: Script adicional con la comparativa de distintos modelos y algunas conclusiones obtenidas (algo más completo)
- **scaler_vgg16.joblib**: Escalador para normalizar los embeddings.
- **pca_vgg16.joblib**: Objeto PCA entrenado para reducción de dimensionalidad.
- **svm_vgg16.joblib**: Modelo SVM entrenado para la predicción final.
- **/data/images/**: Carpeta donde deben colocarse las imágenes a predecir (formato .jpg).


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
