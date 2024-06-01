import os
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import logging
from preprocessing import preprocess_text
from translate import translate_text  # Import the translation function

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)  # Changed to DEBUG for more detailed logs

# Hardcoded labels dictionary
labels_dict = {
    'deepfeedforward_model.h5': ['not_hate_speech', 'offensive_language', 'hate_speech']
}

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

# Load the model and vectorizer
model_path = 'models/deepfeedforward_model.h5'
vectorizer_path = 'models/tfidf_vectorizer.joblib'
model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

st.title('Hate Speech Detection')

# Predefined test texts
test_texts = {
    'not_hate_speech': [
        "I love spending time with my family during the holidays.",  # English
        "आज मौसम बहुत सुहाना है।",  # Hindi: "The weather is beautiful today."
        "Vamos a trabajar juntos para construir una mejor comunidad.",  # Spanish: "Let's work together to build a better community."
        "چھٹیاں گزارنے کے لئے اپنے خاندان کے ساتھ وقت گزارنا مجھے بہت پسند ہے۔"  # Urdu: "I love spending time with my family during the holidays."
    ],
    'offensive_language': [
        "This is a stupid idea.",  # English
        "यह एक बेवकूफी भरा विचार है।",  # Hindi: "This is a stupid idea."
        "¡Qué idiota!",  # Spanish: "What an idiot!"
        "تم کتنے بیوقوف ہو!"  # Urdu: "You are such an idiot."
    ],
    'hate_speech': [
        "I hate people from that country.",  # English
        "मुझे उस देश के लोग पसंद नहीं हैं।",  # Hindi: "I don't like people from that country."
        "Esas personas no son bienvenidas aquí.",  # Spanish: "Those people are not welcome here."
        "واپس جاؤ جہاں سے آئے ہو۔"  # Urdu: "Go back to where you came from."
    ]
}

# Display test options
st.subheader('Test Texts')
category = st.selectbox('Choose a category to test:', ['not_hate_speech', 'offensive_language', 'hate_speech'])
test_text = st.selectbox('Choose a text to evaluate:', test_texts[category])

if st.button('Evaluate Test Text'):
    if test_text:
        if model:
            try:
                logging.debug(f"Evaluating test text: {test_text}")
                translated_text = translate_text(test_text)  # Translate the text
                logging.debug(f"Translated text: {translated_text}")
                preprocessed_text = preprocess_text(translated_text)
                logging.debug(f"Preprocessed text: {preprocessed_text}")
                vectorized_text = vectorizer.transform([preprocessed_text])
                prediction = model.predict(vectorized_text.toarray())
                
                model_name = model_path.split('/')[-1]
                if model_name in labels_dict:
                    labels = labels_dict[model_name]
                    predicted_label = labels[np.argmax(prediction[0])]
                    
                    st.subheader('Results:')
                    st.write(f'Test Text: {test_text}')
                    st.write(f'Translated Text: {translated_text}')
                    st.write(f'Prediction: {predicted_label}')
                    logging.debug(f"Prediction result: {predicted_label}")
                else:
                    st.error(f"Labels not found for model: {model_name}")
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
                translated_text = translate_text(user_input)  # Translate the text
                logging.debug(f"Translated custom text: {translated_text}")
                preprocessed_text = preprocess_text(translated_text)
                logging.debug(f"Preprocessed custom text: {preprocessed_text}")
                vectorized_text = vectorizer.transform([preprocessed_text])
                prediction = model.predict(vectorized_text.toarray())
                
                model_name = model_path.split('/')[-1]
                if model_name in labels_dict:
                    labels = labels_dict[model_name]
                    predicted_label = labels[np.argmax(prediction[0])]
                    
                    st.subheader('Results:')
                    st.write(f'Custom Text: {user_input}')
                    st.write(f'Translated Text: {translated_text}')
                    st.write(f'Prediction: {predicted_label}')
                    logging.debug(f"Custom prediction result: {predicted_label}")
                else:
                    st.error(f"Labels not found for model: {model_name}")
            except Exception as e:
                st.error("An error occurred during text processing or prediction.")
                logging.error(f"Error during text processing or prediction: {e}")
        else:
            st.error('Model not loaded. Please check the model path and file.')
    else:
        st.error('Please enter some text to evaluate.')
st.write(f"Streamlit version: {st.__version__}")
st.write(f"TensorFlow version: {tf.__version__}")
st.write(f"Numpy version: {np.__version__}")
st.write(f"Joblib version: {joblib.__version__}")
st.write(f"Scikit-learn version: {sklearn.__version__}")
st.write(f"Httpcore version: {httpcore.__version__}")
st.write(f"Googletrans version: {googletrans.__version__}")