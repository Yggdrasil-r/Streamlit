import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. Page Config
st.set_page_config(page_title="AI Image Classifier", layout="centered")

# 2. Load the Model
@st.cache_resource
def load_my_model():
    # This looks for the 'my_model.h5' file in your folder
    model_path = 'my_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("Model file 'my_model.h5' not found! Please upload it to the repo.")
        return None

model = load_my_model()

# 3. Define Class Names (CHANGE THESE TO MATCH YOUR MODEL)
CLASS_NAMES = ['Arms Crossed High (Chest)', 
               'Arms Crossed Low (Waist)', 
               'Arms Open and Neutral', 
               'Hands in Pockets ', 
               'One arm crossed']

# 4. User Interface
st.title("🖼️ AI Image Classifier")
st.write("Upload an image, and the AI will tell you what it sees.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Target Image', use_container_width=True)
    
    with st.spinner('Analyzing...'):
        # 5. Preprocessing (matching your 51x51 Colab logic)
        img = image.convert('RGB') # Ensure it's not grayscale/RGBA
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 6. Prediction
        if model is not None:
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Use softmax if model output is raw logits
            
            best_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            # 7. Results
            st.success(f"**Result:** {best_class}")
            st.info(f"**Confidence Level:** {confidence:.2f}%")
