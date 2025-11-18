import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

MODEL_PATH = os.path.join("model", "final_CNN_model.h5")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"softmax_v2": tf.nn.softmax}
)

st.title("Classificador de Dígitos – MNIST")
st.write("Envie uma imagem de um dígito escrito à mão (PNG/JPG).")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Imagem enviada", use_column_width=True)

    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=[0, -1])

    prediction = np.argmax(model.predict(image), axis=1)

    st.success(f"O modelo reconheceu o número: **{prediction[0]}**")
