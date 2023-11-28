import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from statsmodels import api as sm
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


st.title('Affentest')
#st.header('Affentest')
st.markdown('[das Streamlit-Cheat sheet](https://cheat-sheet.streamlit.app/)')
csv_file="affen1trainingsdaten.csv"
df = pd.read_csv(csv_file,delimiter=";")
st.write("Affentest")
#st.dataframe()
st.table(df.iloc[0:10])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

