import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels import api as sm

csv_file="affen1trainingsdaten.csv"
df = pd.read_csv(csv_file)
st.write("Affentest")
