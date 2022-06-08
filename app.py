#  STREAMLIT DEPLOY
#  --- --- --- ---
#  Machine Learning Microbe Prediction:

#import libs
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns

# ______________________
#    DECISION TREE 
#------------------------

microbes_data = 'https://raw.githubusercontent.com/GustavoJannuzzi/StreamlitMicrobePredictor/main/microbes.csv'
base_microbes = pd.read_csv(microbes_data)
base_microbes.drop("Unnamed: 0",axis=1, inplace=True)


# Divisão Previsores e Classe
# X - Previsores
x_microbes = base_microbes.iloc[:, 1:23].values

# Y - Classe
y_microbes = base_microbes.iloc[:, 24].values


#Split the data in for test and training 
from sklearn.model_selection import train_test_split
x_microbes_train, x_microbes_test, y_microbes_train, y_microbes_test = train_test_split(x_microbes, y_microbes, test_size = 0.25, random_state = 0)

# Save the data bases
import pickle
with open ('microbes.pkl', mode = 'wb') as f:
    pickle.dump([x_microbes_train, y_microbes_train, x_microbes_test, y_microbes_test],f)

# Creating the model
from sklearn.tree import DecisionTreeClassifier
import pickle
with open('microbes.pkl','rb') as f:
  x_microbes_train, y_microbes_train, x_microbes_test, y_microbes_test = pickle.load(f)

arvore_microbes = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
arvore_microbes.fit(x_microbes_train, y_microbes_train)

### Create a Pickle file 
import pickle
pickle_out = open("microbes.pkl","wb")
pickle.dump(arvore_microbes, pickle_out)
pickle_out.close()

# ______________________
#       STREAMLIT 
# ------------------------
# Input bar 
col1, col2 = st.columns(2)
with col1:
    Eccentricity = st.number_input("Eccentricity")
    EquivDiameter = st.number_input("EquivDiameter")
    Extrema = st.number_input("Extrema")
    FilledArea = st.number_input("FilledArea")
    Extent = st.number_input("Extent")
    Orientation = st.number_input("Orientation")
    EulerNumber = st.number_input("EulerNumber")
    BoundingBox1 = st.number_input("BoundingBox1")
    BoundingBox2 = st.number_input("BoundingBox2")
    BoundingBox3 = st.number_input("BoundingBox3")
    BoundingBox4 = st.number_input("BoundingBox4")
with col2:
    ConvexHull1 = st.number_input("ConvexHull1")
    ConvexHull2 = st.number_input("ConvexHull2")
    ConvexHull3 = st.number_input("ConvexHull3")
    ConvexHull4 = st.number_input("ConvexHull4")
    MajorAxisLength = st.number_input("MajorAxisLength")
    MinorAxisLength = st.number_input("MinorAxisLength")
    Perimeter = st.number_input("Perimeter")
    ConvexArea = st.number_input("ConvexArea")
    Centroid1 = st.number_input("Centroid1")
    Centroid2 = st.number_input("Centroid2")
    Area = st.number_input("Area")

# If button is pressed
if st.button("Predict"):
    
    # Unpickle classifier
    clf = joblib.load("microbes.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[Eccentricity, EquivDiameter, Extrema,
       FilledArea, Extent, Orientation, EulerNumber, BoundingBox1,
       BoundingBox2, BoundingBox3, BoundingBox4, ConvexHull1,
       ConvexHull2, ConvexHull3, ConvexHull4, MajorAxisLength,
       MinorAxisLength, Perimeter, ConvexArea, Centroid1, Centroid2,
       Area]], 
                     columns = ['Eccentricity', 'EquivDiameter', 'Extrema',
       'FilledArea', 'Extent', 'Orientation', 'EulerNumber', 'BoundingBox1',
       'BoundingBox2', 'BoundingBox3', 'BoundingBox4', 'ConvexHull1',
       'ConvexHull2', 'ConvexHull3', 'ConvexHull4', 'MajorAxisLength',
       'MinorAxisLength', 'Perimeter', 'ConvexArea', 'Centroid1', 'Centroid2',
       'Area'])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.subheader(f"This microbe is a {prediction}")
