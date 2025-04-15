# 📰 verifi.ai

Fake news detection using an ensemble of BERT, GPT-2, and RoBERTa models. Built using HuggingFace Transformers and Streamlit for deployment.

## Features
- BERT-based binary classification
- Dataset: [Kaggle Fake News](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- Streamlit app for web interface

## Dataset
The model is trained on true.csv and fake.csv, zipped and placed in:
- data/raw/true.zip
- data/raw/fake.zip

Each CSV must contain:
- title
- text
- subject
- date

## Features
- Real-time fake news prediction via Streamlit
- Model ensembling
- Zipped CSV handling
- Configurable model paths and parameters

## Setup
```bash
bash setup.sh
```

## Run
```bash
streamlit run app/streamlit_app.py
```

# setup.sh
#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
