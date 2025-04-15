import streamlit as st
from app.utils import run_inference_pipeline

st.set_page_config(page_title="verifi.ai Pro", layout="centered")
st.title("🧠 verifi.ai Pro – Fake News Classifier")
st.markdown("Enter a news headline or story below:")

text_input = st.text_area("Text", height=150)
if st.button("Classify"):
    if text_input.strip():
        prediction = run_inference_pipeline(text_input)
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")