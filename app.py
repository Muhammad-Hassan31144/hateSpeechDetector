import os
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import logging
import re
import string
from spellchecker import SpellChecker
from translate import Translator

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)  # Changed to DEBUG for more detailed logs

# Hardcoded labels dictionary
labels_dict = {
    'deepfeedforward_model.h5': ['Not Hate Speech', 'Offensive Speech', 'Hate Speech']
}

# Initialize spell checker
spell = SpellChecker()

# Function to load models and vectorizers
@st.cache_resource
def load_model_and_vectorizer(model_path, vectorizer_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file not found: {vectorizer_path}")
        return None, None
    
    try:
        model = tf.keras.models.load_model(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        logging.error(f"Error loading model or vectorizer: {e}")
        return None, None

# Text cleaning function
def clean_text(text):
    # Basic text cleaning
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text.lower())
    
    # Spell correction
    corrected_text = ' '.join([spell.correction(word) for word in text.split()])
    return corrected_text

# Load the model and vectorizer
model_path = 'models_MLP/deepfeedforward_model.h5'
vectorizer_path = 'models_MLP/tfidf_vectorizer.joblib'
model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

st.title('Hate Speech Detection')

# Language selection
language_options = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'Urdu': 'ur'}
language = st.selectbox('Choose the language for translation:', list(language_options.keys()))

# Predefined test texts
test_texts = {
    'Not Hate Speech': [
        "It's a beautiful day to go for a walk.",  # English
        "आज का दिन बहुत अच्छा है।",  # Hindi: "Today is a very good day."
        "Estoy emocionado por la fiesta de esta noche.",  # Spanish: "I am excited about the party tonight."
        "آج کا دن بہت اچھا ہے۔"  # Urdu: "Today is a very good day."
    ],
    'Offensive Speech': [
        "You are so dumb.",  # English
        "तुम बहुत बेवकूफ हो।",  # Hindi: "You are very stupid."
        "Eres un idiota.",  # Spanish: "You are an idiot."
        "تم بہت احمق ہو۔"  # Urdu: "You are very stupid."
    ],
    'Hate Speech': [
        "People from that place are not welcome here.",  # English
        "उस जगह के लोग यहाँ स्वागत नहीं हैं।",  # Hindi: "People from that place are not welcome here."
        "Las personas de ese lugar no son bienvenidas aquí.",  # Spanish: "People from that place are not welcome here."
        "اس جگہ کے لوگ یہاں خوش آمدید نہیں ہیں۔"  # Urdu: "People from that place are not welcome here."
    ]
}

# Display test options
st.subheader('Test Texts')
category = st.selectbox('Choose a category to test:', ['Not Hate Speech', 'Offensive Speech', 'Hate Speech'])
test_text = st.selectbox('Choose a text to evaluate:', test_texts[category])

# Store results
# results = []

if st.button('Evaluate Test Text'):
    if test_text:
        if model:
            try:
                logging.debug(f"Evaluating test text: {test_text}")
                translator = Translator(to_lang='en', from_lang=language_options[language])
                translated_text = translator.translate(test_text)
                logging.debug(f"Translated text: {translated_text}")
                cleaned_text = clean_text(translated_text)
                logging.debug(f"Cleaned text: {cleaned_text}")
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text.toarray())
                
                labels = labels_dict.get('deepfeedforward_model.h5', ['Not Hate Speech', 'Offensive Speech', 'Hate Speech'])
                predicted_label = labels[np.argmax(prediction[0])]
                # st.write(f'Predicted Label: {predicted_label}')
                st.subheader('Results:')
                st.write(f'Test Text: {test_text}')
                st.write(f'Translated Text: {translated_text}')
                st.write(f'Prediction: {predicted_label}')
                logging.debug(f"Prediction result: {predicted_label}")
                
                # results.append((test_text, translated_text, predicted_label))
            except Exception as e:
                st.error("An error occurred during text processing or prediction.")
                logging.error(f"Error during text processing or prediction: {e}")
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
            try:
                logging.debug(f"Evaluating custom text: {user_input}")
                translator = Translator(to_lang='en', from_lang=language_options[language])
                translated_text = translator.translate(user_input)
                logging.debug(f"Translated custom text: {translated_text}")
                cleaned_text = clean_text(translated_text)
                logging.debug(f"Cleaned custom text: {cleaned_text}")
                vectorized_text = vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_text.toarray())
                
                labels = labels_dict.get('deepfeedforward_model.h5', ['Not Hate Speech', 'Offensive Speech', 'Hate Speech'])
                predicted_label = labels[np.argmax(prediction[0])]
                st.write(f'Predicted Label: {predicted_label}')
                st.subheader('Results:')
                st.write(f'Custom Text: {user_input}')
                st.write(f'Translated Text: {translated_text}')
                st.write(f'Prediction: {predicted_label}')
                logging.debug(f"Custom prediction result: {predicted_label}")
                
                # results.append((user_input, translated_text, predicted_label))
            except Exception as e:
                st.error("An error occurred during text processing or prediction.")
                logging.error(f"Error during text processing or prediction: {e}")
        else:
            st.error('Model not loaded. Please check the model path and file.')
    else:
        st.error('Please enter some text to evaluate.')

# if st.button('Save Results'):
#     with open('results.txt', 'w') as f:
#         for result in results:
#             f.write(f"Original Text: {result[0]}\n")
#             f.write(f"Translated Text: {result[1]}\n")
#             f.write(f"Prediction: {result[2]}\n")
#             f.write("\n")
#     st.success('Results saved successfully!')
