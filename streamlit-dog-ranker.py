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

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 50px;
        margin-top: 10px;
    }
    .cute-button>button {
        background-color: #ff4b6e;
        color: white;
    }
    .skip-button>button {
        border: 1px solid #gray;
        background-color: white;
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

# Sample dog dataset
DOGS = [
    {
        "id": 1,
        "breed": "Golden Retriever",
        "image_url": "https://images.dog.ceo/breeds/retriever-golden/n02099601_1024.jpg"
    },
    {
        "id": 2,
        "breed": "Corgi",
        "image_url": "https://images.dog.ceo/breeds/corgi-cardigan/n02113186_1030.jpg"
    },
    {
        "id": 3,
        "breed": "Husky",
        "image_url": "https://images.dog.ceo/breeds/husky/n02110185_1469.jpg"
    }
]

def load_and_preprocess_image(image_url):
    """Load and preprocess image for EfficientNet"""
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_cuteness(img_array):
    """Predict cuteness score using EfficientNet"""
    predictions = st.session_state.model.predict(img_array)
    # Using max confidence as a proxy for cuteness
    return float(predictions.max())

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
    
    if st.button("Rate More Dogs"):
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

    # Display current dog
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image(
            current_dog['image_url'],
            caption=current_dog['breed'],
            use_column_width=True
        )

    # Get model prediction
    try:
        img_array = load_and_preprocess_image(current_dog['image_url'])
        cuteness_score = predict_cuteness(img_array)
        st.markdown(f"AI Cuteness Score: {cuteness_score:.2%}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        cuteness_score = 0

    # Rating buttons
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="skip-button">', unsafe_allow_html=True)
        if st.button("👎 Skip"):
            handle_rating(False)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="cute-button">', unsafe_allow_html=True)
        if st.button("❤️ Cute!"):
            handle_rating(True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
