import streamlit as st
import joblib
model =joblib.load('Project_Data')
st.title('Sentiment Analysis')
ip = st.text_input('Enter your message')
op =  model.predict([ip])
if st.button('predict'):
  st.title(op[0])
