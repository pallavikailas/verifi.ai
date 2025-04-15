#!/bin/bash echo "⚙️ Setting up verifi.ai-pro..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio
python download.py

mkdir -p models/bert data/raw data/processed data/embeddings
venv/bin/python scripts/training_pipeline.py

echo "✅ Setup complete. You can now run the app using:" echo " streamlit run app/streamlit_app.py"
