import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inference_pipeline import predict_news

def run_inference_pipeline(text):
    return predict_news(text)

def run_inference_pipeline(text):
    return predict_news(text)
