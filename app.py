import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
import joblib
model =joblib.load('Project_Data')
st.title('Sentiment Analysis')
ip = st.text_input('Enter your message')
op =  model.predict([ip])
if st.button('predict'):
  st.title(op[0])
