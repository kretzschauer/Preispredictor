import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels import api as sm

st.title('Affentest')
st.header('Affentest')
csv_file="affen1trainingsdaten.csv"
df = pd.read_csv(csv_file,delimiter=";")
st.write("Affentest")
#st.dataframe()
st.table(df.iloc[0:10])
