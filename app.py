import streamlit as st
import pandas as  pd
import requests
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
import joblib
model =joblib.load('Project_Data')
st.title('Sentiment Analysis')
ip = []
ip = st.text_input('Enter your message')
op =  model.predict([ip])
if st.button('predict'):
  st.title(op[0])
