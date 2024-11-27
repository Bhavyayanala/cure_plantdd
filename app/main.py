import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Custom CSS
css = """
<style>
    .stApp {
        background-color:#F4F0B9; /* Light yellow background */
    }
    /* Overall body styling */
    body {
        font-family: Arial, sans-serif;
        background-color: #F4F0B9; /* Background color */
        color: #4A4A30;
    }

    /* Header styling */
    .header {
        text-align: left;
        padding: 20px;
        background-color: #424226;
        color: #F8F8E3;
    }

    /* Title styling */
    h1 {
        font-size: 7rem;
        font-weight: bold;
        margin-bottom: 10px;
        font-family: 'Playfair Display', serif;
    }

    /* Upload section styling */
    .upload-section {
        background-color: #f7e2a3; /* Light background color for the upload section */
        color: #4A4A30;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }

    /* Upload button styling */
    .upload-section button {
        background-color: #7C75BA;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 1.2rem;
        cursor: pointer;
        border-radius: 5px;
    }


    /* Link styling */
    a {
        color: #F8F8E3;
        text-decoration: none;
    }

    /* Styling for prediction result */
    .result {
        font-size: 1.2rem;
        font-weight: bold;
        color: #4A4A30;
    }
</style>
"""

# Inject CSS into the app
st.markdown(css, unsafe_allow_html=True)

# Streamlit Title
st.title('Plant Disease Classifier')

# Define layout with two columns
col1, col2 = st.columns([1, 1])  # Adjust the proportions of the columns if needed

# Left column: Upload section
with col1:
    st.markdown('<div class="upload-section"><h2>Upload a Diseased Plant Image</h2></div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    # Display a prediction button if an image is uploaded
    if uploaded_image is not None:
        if st.button('Classify'):
            # Placeholder for model prediction (example)
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')  
            st.markdown(f'<div class="result">Prediction: {prediction}</div>', unsafe_allow_html=True)

# Right column: Display a static or uploaded image
with col2:
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.image("https://image.deondernemer.nl/239479096/feature-crop/1200/630/frank-about-tea-van-de-plantage-rechtstreeks-in-je-brievenbus", caption="Example Image", use_column_width=True)  # Replace with your default image path

