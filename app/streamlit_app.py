import streamlit as st
from app.utils import run_inference_pipeline

st.set_page_config(page_title="verifi.ai")
st.title("💡 verifi.ai")
st.write("Enter a news headline or article:")

text_input = st.text_area("Text", height=200)
if st.button("Predict"):
    if text_input:
        label = run_inference_pipeline(text_input)
        st.success(f"Prediction: {label}")
    else:
        st.warning("Please enter text before predicting.")
