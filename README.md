# Análisis de Sentimiento de Reseñas de Películas con BiLSTM

**Autores:** Daniel Brand Taborda, Jhonier Raúl Jiménez
**Curso:** Deep Learning - 2025

**Enlace del video:** https://youtu.be/iSqDng-TPaw

---

## 📝 Descripción del Proyecto

Este repositorio contiene la solución para un proyecto de análisis de sentimiento, cuyo objetivo es clasificar reseñas de películas del famoso **dataset IMDB** como **positivas** o **negativas**.

Para lograrlo, se implementó un modelo de Deep Learning basado en una **Red Neuronal Recurrente Bidireccional (BiLSTM)**. El proyecto abarca desde el análisis exploratorio de los datos y su preprocesamiento (limpieza, lematización, tokenización) hasta el entrenamiento y la evaluación rigurosa del modelo.

---

## 🎥 Video de Presentación

En el siguiente video se encuentra la presentación ejecutiva del proyecto, explicando la metodología, la arquitectura y los resultados obtenidos.

👉 **Enlace al video:** **[AQUÍ DEBES PEGAR EL ENLACE A TU VIDEO DE YOUTUBE]**

---

## 📊 Resultados

El modelo final, tras ser entrenado durante 5 épocas, alcanzó un rendimiento sólido y balanceado en el conjunto de prueba (datos nunca antes vistos):

* **Exactitud (Accuracy) General:** **86.6%**
* **F1-Score (Promedio):** **87%**

---

## 🗂️ Estructura del Repositorio

El proyecto está organizado de manera modular para facilitar su comprensión y reproducibilidad.

* **/data**: Contiene el dataset original `IMDB Dataset.csv`.
* **/processed_data**: Contiene los datos de entrenamiento y prueba ya limpios y preprocesados, listos para ser consumidos por el modelo.
* `01 - exploración de datos.ipynb`: Notebook para el Análisis Exploratorio de Datos (EDA).
* `02 - preprocesado.ipynb`: Notebook que realiza toda la limpieza y preparación de los datos de texto.
* `03 - arquitectura de linea de base.ipynb`: Notebook donde se define la arquitectura del modelo BiLSTM y se preparan las secuencias (tokenización y padding).
* `04 - entrenamiento y evaluacion.ipynb`: Notebook para entrenar, evaluar el modelo y visualizar los resultados.
* `sentiment_model.h5`: El modelo final entrenado y guardado.
* `tokenizer_config.json`: El objeto Tokenizer de Keras guardado, esencial para procesar nuevas reseñas con el mismo vocabulario.
* `requirements.txt`: Archivo con las dependencias exactas del proyecto.
* `INFORME_PROYECTO.PDF`: El informe ejecutivo detallado del proyecto.
* `ENTREGA1.PDF`: El informe de la primera entrega del proyecto.

---

## 🚀 Cómo Ejecutar el Proyecto

Para replicar los resultados, puedes seguir estos pasos.

### Prerrequisitos

* Python 3.8 o superior
* Pip (manejador de paquetes de Python)

### Instalación en un Entorno Local

1.  **Clona el repositorio:**
    ```bash
    git clone [URL-DE-TU-REPOSITORIO]
    cd [NOMBRE-DEL-REPOSITORIO]
    ```

2.  **Crea y activa un entorno virtual (recomendado):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    El archivo `requirements.txt` contiene todas las librerías necesarias con sus versiones exactas. Instálalas con el siguiente comando:
    ```bash
    pip install -r requirements.txt
    ```
    Las dependencias clave son:
    - `tensorflow==2.19.0`
    - `pandas==2.3.0`
    - `numpy==2.3.1`
    - `scikit-learn==1.7.0`
    - `nltk==3.9.1`
    - `matplotlib==3.10.3`
    - `seaborn==0.13.2`

    *Nota: La primera vez que ejecutes el notebook de preprocesado, NLTK podría necesitar descargar componentes adicionales como `wordnet` y `stopwords`.*

4.  **Ejecuta los notebooks en orden:**
    Abre Jupyter Notebook o Jupyter Lab y ejecuta los notebooks en el siguiente orden numérico para replicar el pipeline completo:
    1.  `01 - exploración de datos.ipynb`
    2.  `02 - preprocesado.ipynb`
    3.  `03 - arquitectura de linea de base.ipynb`
    4.  `04 - entrenamiento y evaluacion.ipynb`

### Ejecución en Google Colab

Alternativamente, los notebooks están diseñados para ser **directamente reproducibles en Google Colab**. Simplemente sube la carpeta del proyecto a tu Google Drive, o clona el repositorio directamente en una celda de Colab, y ejecuta los notebooks en orden.

---

## 💾 Dataset

El conjunto de datos utilizado es el **"IMDB Dataset of 50K Movie Reviews"**, disponible públicamente en Kaggle.

* **Enlace:** [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python
* **Deep Learning:** TensorFlow / Keras
* **Procesamiento de Lenguaje Natural:** NLTK
* **Manipulación de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Entorno:** Jupyter Notebooks, Google Colab

---

¡Gracias por revisar el proyecto!