import streamlit as st
import pickle

# Load the model
with open('KNeighbors_model.pkl', 'rb') as file:
    knn = pickle.load(file)

# Streamlit App Layout
st.title("Iris Flower Prediction")

# Input fields for the features
Id = st.number_input("Id", min_value=0, step=1)
sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

# Predict button
if st.button('Predict'):
    # Make prediction
    result = knn.predict([[Id, sepal_length, sepal_width, petal_length, petal_width]])[0]
    
    # Display the result
    st.write(f"The predicted class is: {result}")
