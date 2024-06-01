# Hate Speech Detection Web Application

This repository contains a web application for detecting hate speech using a deep feedforward neural network. The application allows users to input text and receive a prediction indicating whether the text is classified as hate speech, offensive language, or not hate speech. The model was trained as part of an assignment, but leveraging web development skills, I developed a web app to showcase the trained model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Challenges Solved](#challenges-solved)
- [Files](#files)
- [Acknowledgments](#acknowledgments)

## Overview

The goal of this project was to train a deep feedforward neural network to detect hate speech in text. The trained model is deployed in a web application built using Streamlit. Users can enter text, and the app will classify it into one of three categories: not hate speech, offensive language, or hate speech.

## Features

- **Text Classification**: Classifies input text into three categories.
- **Predefined Test Texts**: Evaluate predefined texts from each category.
- **Custom Text Evaluation**: Allows users to input custom text for evaluation.
- **Interactive Interface**: User-friendly interface built with Streamlit.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/hate-speech-detection-webapp.git
    cd hate-speech-detection-webapp
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the trained models (`deepfeedforward_model.h5` and `tfidf_vectorizer.joblib`) in the `models` directory.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open the provided URL in your browser to access the web application.

## Challenges Solved

1. **Model Training**: Trained a deep feedforward neural network to classify text into hate speech, offensive language, and not hate speech categories.
2. **Text Preprocessing**: Implemented a robust text preprocessing pipeline that includes tokenization, lemmatization, and removal of stopwords.
3. **Model Deployment**: Successfully deployed the trained model in a web application using Streamlit.
4. **User Interface**: Created an interactive and user-friendly interface for the model, allowing both predefined and custom text evaluation.

## Files

- `app.py`: Main application script for the web app.
- `preprocessing.py`: Contains the text preprocessing functions.
- `requirements.txt`: List of required Python packages.
- `models/`: Directory containing the trained model (`deepfeedforward_model.h5`) and the TF-IDF vectorizer (`tfidf_vectorizer.joblib`).

## Acknowledgments

This project was developed as part of a coursework assignment to train a deep feedforward neural network model. But By leveraging my web development skills, I extended the project to include a web application for showcasing the trained model.

Special thanks to my instructors for their support and guidance throughout this project.

