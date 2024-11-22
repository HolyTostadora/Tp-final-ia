import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Cargar el modelo
model = tf.keras.models.load_model('model.h5')

# Configuración de Streamlit
st.title("Clasificador de Imágenes")
st.write("Sube una imagen para predecir a qué categoría pertenece.")

# Subir la imagen
uploaded_image = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen subida", use_column_width=True)
    st.write("")
    
    # Preprocesar la imagen para que coincida con el formato de entrada del modelo
    image = image.resize((224, 224))  # Redimensionar si es necesario
    image = np.array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Añadir dimensión para el batch

    # Hacer predicción
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)  # La categoría con mayor probabilidad

    # Mostrar la categoría predicha
    categories = ['Categoria 1', 'Categoria 2', 'Categoria 3', 'Categoria 4', 'Categoria 5', 'Categoria 6', 'Categoria 7']
    predicted_category = categories[predicted_class[0]]

    st.write(f"Predicción: {predicted_category}")
