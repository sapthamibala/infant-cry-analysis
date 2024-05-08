import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from streamlit_option_menu import option_menu
# Load the trained model
model = load_model("final.h5")

# Define the number of classes
num_classes = 5

# Function to preprocess the input audio, convert it into a spectrogram image, and make predictions
def predict_cry_type(audio_file):
    # Load the audio file
    audio_data, sr = librosa.load(audio_file, sr=None)

    # Compute the mel spectrogram
    ms = librosa.feature.melspectrogram(y=audio_data, sr=sr)

    # Convert to decibels
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # Plot the spectrogram and save as an image
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig("temp_spec.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    st.image("temp_spec.png", caption='Spectrogram Image', use_column_width=True)
    # Preprocess the image
    img = image.load_img("temp_spec.png", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255

    # Make prediction
    prediction = model.predict(img_array)
    class_labels = ['Belly Pain', 'Burping', 'Discomfort', 'Hungry', 'Tired']
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class



# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Pages", ["Home", 'About Page'], 
        icons=['house', 'gear'], menu_icon="copy", default_index=0)

# Streamlit UI
if selected=='Home':
    st.title("Infant Cry Type Detection👶...")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        cry_type = predict_cry_type(uploaded_file)
        st.markdown('---')
        st.write("### Predicted Baby Cry Type:", cry_type)
        st.markdown('---')

if selected=='About Page':
    st.markdown('''# About Infant Cry Type Detection

Welcome to Infant Cry Type Detection, an innovative system designed to classify different types of infant cries based on their spectrogram patterns.

---
                
## Purpose
Infant crying is one of the primary means of communication for babies, indicating various needs such as hunger, discomfort, pain, or tiredness. However, for parents and caregivers, deciphering the underlying cause of a baby's cry can often be challenging. The Infant Cry Type Detection system aims to alleviate this challenge by automatically classifying different types of cries, providing valuable insights to parents and caregivers.

---

## How It Works
The system utilizes advanced machine learning techniques to analyze the acoustic features of infant cries. Here's an overview of the system architecture:

- **Data Collection**: Audio recordings of infant cries are collected from various sources, ensuring a diverse and representative dataset.
  
- **Preprocessing**: The audio recordings undergo preprocessing steps, including resampling, noise reduction, and feature extraction. Librosa, a Python library for audio and music analysis, is used for these tasks.

- **Feature Extraction**: Spectrogram images are generated from the preprocessed audio data using the Mel-Frequency Cepstral Coefficients (MFCCs) technique. These spectrograms capture the frequency content of the cries over time.

- **Model Training**: A deep learning model, implemented using TensorFlow and Keras, is trained on the spectrogram images. The model architecture typically consists of convolutional neural network (CNN) layers followed by dense layers for classification.

- **Model Evaluation**: The trained model is evaluated using a separate validation dataset to assess its performance in classifying different cry types accurately.

- **Deployment**: The trained model is deployed as a web application using Streamlit, a Python framework for building interactive web applications. Users can upload audio files of infant cries, and the system predicts the cry type based on the spectrogram analysis.

---

## Our Mission
Our mission is to empower parents and caregivers with valuable insights into their baby's needs, enhancing the quality of care and nurturing experiences for infants and their families.

---

## Contact Us
If you have any questions, feedback, or suggestions, please feel free to reach out to us at [subikshasapthami@gmail.com](mailto:subikshasapthami@gmail.com). We'd love to hear from you!
''')