from tensorflow.keras.models import load_model
from utils.utils import Utils
import streamlit as st
import time

# Load the IMDB dataset word index
helper = Utils()

# Load the model
model = load_model('model.h5')

# Prediction Functions
def predict_sentiments(review):
    preprocessed_input = helper.preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    return prediction[0][0]

# Streamlit App
st.title("IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review to classify it as positive or negative.")

# user input
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    message = st.empty()
    
    if user_input != "":    
        prediction_score = predict_sentiments(user_input)
        if prediction_score > 0.5:
            message.success("Positive Sentiment Detected!")
        elif prediction_score < 0.5:
            message.error("Negative Sentiment Detected!")
        else:
            message.info("Neutral Sentiment Detected!")
    else:
        message.info('Please enter a movie review!')
    
    time.sleep(2)
    message.empty()