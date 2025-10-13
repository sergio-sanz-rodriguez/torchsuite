"""
Provides utility functions for deep learning workflows in PyTorch.  
Includes dataset handling, visualization, model saving/loading, evaluation metrics, and statistical analysis.  

Functions:
- **File & Directory Management**: `walk_through_dir`, `zip_folder`, `download_data`
- **Visualization**: `plot_decision_boundary`, `plot_predictions`, `display_random_images`, `pred_and_plot_image`, `pred_and_plot_image_imagenet`, `plot_loss_curves`, `plot_confusion_matrix`, `plot_class_distribution` 
- **Training & Evaluation**: `print_train_time`, `save_model`, `load_model`, `accuracy_fn`, `get_most_wrong_examples`
- **Reproducibility**: `set_seeds`
- **ROC & AUC Analysis**: `find_roc_threshold_tpr`, `find_roc_threshold_fpr`, `find_roc_threshold_f1`, `find_roc_threshold_accuracy`, `partial_auc_score`, `cross_val_partial_auc_score`
"""

import torch
import matplotlib.pyplot as plt
import os
import zipfile
import requests
import gdown
from pathlib import Path


# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
def walk_through_dir(dir_path):

    """
    Walks through dir_path returning its contents.
    Args:
        dir_path (str): target directory

    Returns:
        A print out of:
            number of subdiretories in dir_path
            number of images (files) in each subdirectory
            name of each subdirectory
    """

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")



# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    
    """
    Plots linear training data and test data and compares predictions.
    """

    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


def print_train_time(start, end, device=None):
    
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """

    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

def set_seeds(seed: int=42):

    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """

    # Set the seed for general torch operations
    torch.manual_seed(seed)

    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """
    Downloads a zipped dataset from source and unzips to destination.
    Supports direct URLs and Google Drive links.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")

    """

    destination = Path(destination)

    # Skip if already exists
    if destination.exists() and any(destination.iterdir()):
        print(f"[INFO] {destination} already exists and is not empty, skipping download.")
        return destination

    print(f"[INFO] Creating destination directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    # Always use a clean target filename to avoid invalid characters
    target_file = destination / "downloaded_data.zip"

    # --- Handle Google Drive links ---
    if "drive.google.com" in source or "usercontent.google.com" in source:
        if gdown is None:
            raise ImportError("Install 'gdown' to download from Google Drive: pip install gdown")
        print(f"[INFO] Downloading from Google Drive: {source}")
        gdown.download(url=source, output=str(target_file), quiet=False)
    else:
        print(f"[INFO] Downloading from {source}")
        response = requests.get(source, allow_redirects=True)
        response.raise_for_status()
        with open(target_file, "wb") as f:
            f.write(response.content)

    # --- Unzip downloaded file ---
    print(f"[INFO] Unzipping {target_file} ...")
    try:
        with zipfile.ZipFile(target_file, "r") as zip_ref:
            zip_ref.extractall(destination)
    except zipfile.BadZipFile:
        raise RuntimeError(f"The file {target_file} is not a valid ZIP archive.")

    if remove_source:
        target_file.unlink()
        print(f"[INFO] Removed zip file: {target_file.name}")

    print(f"[INFO] Data ready at {destination}")
    return destination



def download_data_(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    

def load_model(model: torch.nn.Module,
               model_weights_dir: str,
               model_weights_name: str):
               #hidden_units: int):

    """Loads a PyTorch model from a target directory.

    Args:
    model: A target PyTorch model to load.
    model_weights_dir: A directory where the model is located.
    model_weights_name: The name of the model to load.
      Should include either ".pth" or ".pt" as the file extension.

    Example usage:
    model = load_model(model=model,
                       model_weights_dir="models",
                       model_weights_name="05_going_modular_tingvgg_model.pth")

    Returns:
        The loaded PyTorch model.
    """

    # Create the model directory path
    model_dir_path = Path(model_weights_dir)

    # Create the model path
    assert model_weights_name.endswith(".pth") or model_weights_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = model_dir_path / model_weights_name

    # Load the model
    print(f"[INFO] Loading model from: {model_path}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model


def zip_folder(folder_path, output_zip, exclusions):

    """
    Zips the contents of a folder, excluding specified files or folders.
    folder_to_zip = "demos/foodvision_mini"  # Change this to your folder path
    output_zip_file = "demos/foodvision_mini.zip"
    exclusions = ["__pycache__", "ipynb_checkpoints", ".pyc", ".ipynb"]
    """

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if all(excl not in os.path.join(root, d) for excl in exclusions)]
            for file in files:
                file_path = os.path.join(root, file)
                # Skip excluded files
                if any(excl in file_path for excl in exclusions):
                    continue
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname=arcname)