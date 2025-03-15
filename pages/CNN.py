import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image
from PIL import Image

classPred = ['Dog','Cat']

# Load the trained model
model = tensorflow.keras.models.load_model("cnn nueral/CNN_MODEL.h5")

# Streamlit UI
st.title("CNN Cat and Dog Classifier")
st.write("Upload an image and let the model predict if it's cat or dog!")
st.write("Please ensure that the picure is indeed a cat or a dog. Otherwise it might show some interesting result. XD")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    firstImg = Image.open(uploaded_file).convert("RGB")

    # Resize to match model input (32x32)
    img = firstImg.resize((32, 32))

    # Convert to NumPy array
    img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_array.astype(np.float32), cv2.COLOR_RGB2GRAY)

    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    img_gray = np.expand_dims(img_gray, axis=-1)  # Add channel dimension
    img_gray = np.expand_dims(img_gray, axis=0)   # Add batch dimension

    # Display the uploaded grayscale image

    # Predict using the model
    y_pred = model.predict(img_gray)
    predicted_class = np.argmax(y_pred, axis=1)[0]


    # Display prediction result
    if predicted_class <= 1:
        st.write("# Predicted Class: {}".format(classPred[predicted_class]))
        st.image(firstImg, caption="Uploaded Image")
    else :
        st.write("# Not a dog nor cat")
        st.image(firstImg, caption="Uploaded Image")
