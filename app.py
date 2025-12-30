import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn.h5')


#Function to preprocess review
def preprocess_review(review):
    review = review.lower().split()
    encode_review = [ word_index.get(word,2)+3 for word in review] ## find the index of that word
    encode_review = sequence.pad_sequences([encode_review], maxlen=500)
    return encode_review

## prediction Function
def predict_sentiment(review):
  preprocess_input = preprocess_review(review)
  prediction = model.predict(preprocess_input)
  if prediction > 0.5:
    return "Positive", prediction[0][0]
  else:
    return "Negative", prediction[0][0]
  

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")
user_input = st.text_area("Movie Review", height=200)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.write("Please enter a valid movie review.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: **{confidence:.4f}**")