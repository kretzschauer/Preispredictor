import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels import api as sm

csv_file="pizza_dataset_relative_price.csv"
df = pd.read_csv(csv_file)
Y = df['Relative Price']
X = df.drop(['Relative Price', 'Pizza Name', 'Topping 3_Meat', 'Topping 3_None', 'Topping 4_Fish', 'Topping 4_None',
             'Overall Weight'], axis=1) 
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
model = sm.OLS(y_train, X_train).fit()
st.write("model created")
