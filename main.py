import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO

st.title('Malaria Detection Model In Image blood smear using Deep Learning ')

class_names = ['Parasite', 'Uninfected']
load_image = st.file_uploader('Your Image here')
model = tf.keras.models.load_model('models/1')


button = st.button('Predict')

if button:

    # try:
    # st.write(load_image)
    st.write(load_image)
    pic = Image.open(load_image)
    pic = pic.resize((256, 256))
    pic_array = np.array(pic)
    pic_arrays = np.expand_dims(pic_array, 0)
    pic_array = pic_array.reshape((1, 256, 256, 3))


    prediction = class_names[np.argmax(model.predict(pic_arrays))]

    st.write(f'The image was predicted to be {prediction}')
     # except:
     #    wt.write('No image was selected')
