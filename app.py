import streamlit as st
st.title("Material Detector")
uploaded_file = st.file_uploader("Upload an image")
if uploaded_file:
    st.image(uploaded_file)
    st.write("Prediction: Plastic")  # Example
