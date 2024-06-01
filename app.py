import os
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib  # Ensure joblib is imported
from preprocessing import preprocess_text

# Hardcoded labels dictionary
labels_dict = {
    'deepfeedforward_model.h5': ['not_hate_speech', 'offensive_language', 'hate_speech']  # Update labels to match your model output
}

# Load the deep feedforward model
model_path = 'models/deepfeedforward_model.h5'
vectorizer_path = 'models/tfidf_vectorizer.joblib'  # Assuming you saved the vectorizer

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Successfully loaded model: {model_path}")
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
else:
    st.error(f"Model file not found: {model_path}")

# Load the TF-IDF vectorizer
if os.path.exists(vectorizer_path):
    try:
        vectorizer = joblib.load(vectorizer_path)
    except Exception as e:
        st.error(f"Error loading vectorizer {vectorizer_path}: {e}")
else:
    st.error(f"Vectorizer file not found: {vectorizer_path}")

st.title('Hate Speech Detection')

# Predefined test texts
test_texts = {
    'not_hate_speech': [
        "I love spending time with my family during the holidays.",
        "The weather is beautiful today.",
        "Let's work together to build a better community."
    ],
    'offensive_language': [
        "This is a stupid idea.",
        "What a moron!",
        "You are such an idiot."
    ],
    'hate_speech': [
        "I hate people from that country.",
        "Those people are not welcome here.",
        "Go back to where you came from."
    ]
}

# Display test options
st.subheader('Test Texts')
category = st.selectbox('Choose a category to test:', ['not_hate_speech', 'offensive_language', 'hate_speech'])
test_text = st.selectbox('Choose a text to evaluate:', test_texts[category])

if st.button('Evaluate Test Text'):
    if test_text:
        if model:
            preprocessed_text = preprocess_text(test_text)
            vectorized_text = vectorizer.transform([preprocessed_text])  # Transform using vectorizer
            prediction = model.predict(vectorized_text.toarray())  # Ensure input is in array format
            
            model_name = model_path.split('/')[-1]
            if model_name in labels_dict:
                labels = labels_dict[model_name]
                
                predicted_label = labels[np.argmax(prediction[0])]
                
                st.subheader('Results:')
                st.write(f'Test Text: {test_text}')
                st.write(f'Prediction: {predicted_label}')
            else:
                st.error(f"Labels not found for model: {model_name}")
        else:
            st.error('Model not loaded. Please check the model path and file.')
    else:
        st.error('Please choose a text to evaluate.')

# User input
st.subheader('Custom Text Evaluation')
user_input = st.text_area('Enter text to evaluate:', '')

if st.button('Evaluate Custom Text'):
    if user_input:
        if model:
            preprocessed_text = preprocess_text(user_input)
            vectorized_text = vectorizer.transform([preprocessed_text])  # Transform using vectorizer
            prediction = model.predict(vectorized_text.toarray())  # Ensure input is in array format
            
            model_name = model_path.split('/')[-1]
            if model_name in labels_dict:
                labels = labels_dict[model_name]
                
                predicted_label = labels[np.argmax(prediction[0])]
                
                st.subheader('Results:')
                st.write(f'Custom Text: {user_input}')
                st.write(f'Prediction: {predicted_label}')
            else:
                st.error(f"Labels not found for model: {model_name}")
        else:
            st.error('Model not loaded. Please check the model path and file.')
    else:
        st.error('Please enter some text to evaluate.')
