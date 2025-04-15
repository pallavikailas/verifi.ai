import zipfile
import os

def extract_zip(zip_path, extract_to='data/raw/'):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found!")

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"✅ Extracted to {extract_to}")

# Example usage
extract_zip('data/raw/Fake.csv.zip')
extract_zip('data/raw/True.csv.zip')
