import os
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
from models import feature_extractor

def download_kaggle_dataset(dataset_name: str, download_path: str):
    """
    Download a dataset from Kaggle.

    Args:
        dataset_name: Kaggle dataset handle (e.g., "owner/dataset")
        download_path: Directory path where the dataset will be downloaded
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading dataset: {dataset_name} to {download_path}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset downloaded successfully to {download_path}")


def main():
    print("Hello from deep-learning-based-signature-forgery-detection-for-personal-identity-authentication!")
    
    # Download the BHSIG260 dataset
    print("\n=== Downloading BHSig260 Dataset ===")
    download_kaggle_dataset("nth2165/bhsig260-hindi-bengali", "data")
    
    # Download the CEDAR dataset
    print("\n=== Downloading CEDAR Dataset ===")
    download_kaggle_dataset("shreelakshmigp/cedardataset", "data/cedardataset")
    
    print("\nAll datasets downloaded successfully!")
    
    # Initialize sample model
    print("\n=== Initializing Sample Model ===")
    model = feature_extractor.DenseNetFeatureExtractor()
    print(model)
    print("Model initialized successfully!")
    
    # Prepare split ratios
    print("\n=== Preparing Split Ratios ===")
    data_root = "data"
    output_dir = "data/ratio_splits"
    
    prepare_script = os.path.join("scripts", "prepare_split_ratios.py")
    cmd = [
        "python",
        prepare_script,
        "--data_root", data_root,
        "--output_dir", output_dir,
        "--seed", "42",
        "--ratios", "65:18:18,70:15:15,60:20:20"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\nSplit ratios prepared successfully!")
    else:
        print(f"\nWarning: prepare_split_ratios.py exited with code {result.returncode}")

if __name__ == "__main__":
    main()
