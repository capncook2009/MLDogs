import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Cute Dog Ranker",
    page_icon="🐕",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 50px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'ratings' not in st.session_state:
    st.session_state.ratings = {}
if 'model' not in st.session_state:
    st.session_state.model = EfficientNetB0(weights='imagenet', include_top=True)

# Using more reliable image URLs
DOGS = [
    {
        "id": 1,
        "breed": "Golden Retriever",
        "image_url": "https://raw.githubusercontent.com/jigsawpieces/dog-api-images/main/golden-retriever/n02099601_1024.jpg"
    },
    {
        "id": 2,
        "breed": "Corgi",
        "image_url": "https://raw.githubusercontent.com/jigsawpieces/dog-api-images/main/cardigan-corgi/n02113186_1030.jpg"
    },
    {
        "id": 3,
        "breed": "Husky",
        "image_url": "https://raw.githubusercontent.com/jigsawpieces/dog-api-images/main/husky/n02110185_1469.jpg"
    }
]

def load_and_preprocess_image(image_url):
    """Load and preprocess image for EfficientNet with better error handling"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Open image with explicit format detection
        image_data = BytesIO(response.content)
        img = Image.open(image_data).convert('RGB')  # Convert to RGB to ensure compatibility
        
        # Resize with antialiasing
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        img_array = np.array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch image: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def predict_cuteness(img_array):
    """Predict cuteness score using EfficientNet"""
    if img_array is None:
        return 0.0
    try:
        predictions = st.session_state.model.predict(img_array, verbose=0)
        # Using top-5 average confidence as cuteness score
        top_5_mean = np.mean(np.sort(predictions[0])[-5:])
        return float(top_5_mean)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.0

def handle_rating(is_cute):
    """Handle user rating and move to next dog"""
    current_dog = DOGS[st.session_state.current_index]
    st.session_state.ratings[current_dog['id']] = is_cute
    
    if st.session_state.current_index < len(DOGS) - 1:
        st.session_state.current_index += 1
    st.experimental_rerun()

def show_results():
    """Display rating results"""
    cute_count = sum(1 for r in st.session_state.ratings.values() if r)
    
    st.markdown("## 🎉 Rating Complete!")
    st.markdown(f"### You found {cute_count} out of {len(DOGS)} dogs cute!")
    
    # Display ratings summary
    results_df = pd.DataFrame([
        {
            'Breed': DOGS[i]['breed'],
            'Rating': '❤️' if st.session_state.ratings.get(DOGS[i]['id'], False) else '⏭️'
        }
        for i in range(len(DOGS))
    ])
    st.dataframe(results_df, hide_index=True)
    
    if st.button("Rate More Dogs", key='restart'):
        st.session_state.current_index = 0
        st.session_state.ratings = {}
        st.experimental_rerun()

def main():
    st.title("🐕 Cute Dog Ranker")
    st.markdown("### Rate these adorable dogs!")

    # Show results if all dogs rated
    if len(st.session_state.ratings) == len(DOGS):
        show_results()
        return

    current_dog = DOGS[st.session_state.current_index]
    
    # Display progress
    st.progress((st.session_state.current_index) / len(DOGS))
    st.markdown(f"Dog {st.session_state.current_index + 1} of {len(DOGS)}")

    # Display current dog with error handling
    try:
        response = requests.get(current_dog['image_url'])
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=current_dog['breed'], use_column_width=True)
        
        # Get model prediction
        img_array = load_and_preprocess_image(current_dog['image_url'])
        if img_array is not None:
            cuteness_score = predict_cuteness(img_array)
            st.markdown(f"AI Cuteness Score: {cuteness_score:.2%}")
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.markdown("⚠️ Image temporarily unavailable")

    # Rating buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👎 Skip", key=f"skip_{current_dog['id']}"):
            handle_rating(False)
    
    with col2:
        if st.button("❤️ Cute!", key=f"cute_{current_dog['id']}"):
            handle_rating(True)

if __name__ == "__main__":
    main()
