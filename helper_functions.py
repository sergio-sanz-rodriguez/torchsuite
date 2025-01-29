import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import nn
import os
import zipfile
from pathlib import Path
import requests
import torchvision
from typing import List, Dict, Tuple
import random
from PIL import Image
from torchvision.transforms import v2
from tqdm.auto import tqdm
from tkinter import Tk
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score, roc_curve


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

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

NUM_WORKERS = os.cpu_count()


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


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image_imagenet(model: torch.nn.Module,
                                 image_path: str, 
                                 class_names: List[str],
                                 image_size: Tuple[int, int] = (224, 224),
                                 transform: torchvision.transforms = None,
                                 device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        image_path (str): A file path to the image being predicted on. 
        class_names (List[str]): A list of class names such as ["pizza", "steak", "sushi"].
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # 0. Make sure the model is on the target device
    model.to(device) 
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = v2.Compose([
            v2.Resize(image_size),                                  # 1. Reshape all images to image_size
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),    # 2. convert to tensor and normalize
            v2.Normalize(mean=[0.485, 0.456, 0.406],                # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225])                 # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
                         ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)



def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

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

def display_random_images(dataset: torch.utils.data.dataset.Dataset, # or torchvision.datasets.ImageFolder?
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          rows: int = 5,
                          cols: int = 5,
                          seed: int = None):
    
   
    """Displays a number of random images from a given dataset.

    Args:
        dataset (torch.utils.data.dataset.Dataset): Dataset to select random images from.
        classes (List[str], optional): Names of the classes. Defaults to None.
        n (int, optional): Number of images to display. Defaults to 10.
        display_shape (bool, optional): Whether to display the shape of the image tensors. Defaults to True.
        rows: number of rows of the subplot
        cols: number of columns of the subplot
        seed (int, optional): The seed to set before drawing random images. Defaults to None.
    
    Usage:
    display_random_images(train_data, 
                      n=16, 
                      classes=class_names,
                      rows=4,
                      cols=4,
                      display_shape=False,
                      seed=None)
    """

    # 1. Setup the range to select images
    n = min(n, len(dataset))
    # 2. Adjust display if n too high
    if n > rows*cols:
        n = rows*cols
        #display_shape = False
        print(f"For display purposes, n shouldn't be larger than {rows*cols}, setting to {n} and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(cols*4, rows*4))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(rows, cols, i+1)        
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)


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


def download_data(source: str, 
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

def get_most_wrong_examples(model: torch.nn.Module,
                            test_dataloader: torch.utils.data.DataLoader,
                            num_samples:int=5,
                            plot_images:bool=True,
                            n_cols:int=5,
                            title_font_size:int=20,
                            device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
                            ):
    '''
    Returns the most wrong examples from a trained model
    Args:
        model: a trained model
        test_dataloader: a test dataloader
        num_samples: number of samples to return
        plot_images: whether to plot the images
        device: device to use for the model
    '''

    # Make predictions on test dataset
    y_preds = []
    y_probs = []

    model.eval()
    model.to(device)

    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), colour='#FF9E2C', desc="Making predictions"):

            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            
            # Do the forward pass
            y_logit = model(X)

            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            y_prob = torch.softmax(y_logit, dim=1).max(dim=1)[0]
            
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
            y_probs.append(y_prob.cpu())

    # Concatenate list of predictions into a tensor
    test_pred_tensor = torch.cat(y_preds)
    test_prob_tensor = torch.cat(y_probs)

    # Convert tensors to lists
    y_preds = test_pred_tensor.tolist()
    y_probs = test_prob_tensor.tolist()

    sample = [i[0] for i in test_dataloader.dataset.imgs]
    label = [i[1] for i in test_dataloader.dataset.imgs]
    prediction = y_preds
    prob = y_probs

    #len(sample), len(label), len(prediction), len(prob)
    df = pd.DataFrame({'sample': sample, 'label': label, 'prediction': prediction, 'prob': prob})
    df['match'] = df['label'] == df['prediction']
    df_wrong = df[df['match'] == False]
    df_wrong_sorted = df_wrong.sort_values(by='prob', ascending=False)
    df_top_wrong = df_wrong_sorted.head(num_samples)
    df_top_wrong.reset_index(inplace=True)

    num_samples = df_top_wrong.shape[0]

    if plot_images or num_samples == 0:
                  
        class_names = test_dataloader.dataset.classes

        n_rows = int(num_samples/n_cols) + 1

        root = Tk()
        screen_width_px = root.winfo_screenwidth()
        dpi = plt.rcParams['figure.dpi']
        screen_width_in = screen_width_px / dpi

        _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(screen_width_in, screen_width_in*n_rows // n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i*n_cols+j 
                if idx >= num_samples:
                    axes[i, j].set_visible(False)
                else:
                    img = plt.imread(df_top_wrong.loc[idx, 'sample'])
                    axes[i, j].imshow(img)
                    predicted = df_top_wrong.loc[idx, 'prediction']
                    predicted_class = class_names[predicted]
                    actual = df_top_wrong.loc[idx, 'label']
                    actual_class = class_names[actual]
                    axes[i, j].set_title(f"Pred: {predicted_class} (Act: {actual_class})", fontsize=title_font_size)
                    axes[i, j].axis('off')

    return df_top_wrong

def zip_folder(folder_path, output_zip, exclusions):

    """Zips the contents of a folder, excluding specified files or folders.
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


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def find_roc_threshold_tpr(y, y_pred, value_target):
    """
    This function calculates the threshold and false positive rate corresponding to a true positive rate of value_target (from 0 to 1).
       
    y                     # Real labels
    y_pred                # Predicted labels
    value_target          # False positive rate value
    
    Returns:
    
    threshold             # Threshold value
    true_positive_rate   # True positive rate value
    """

    fpr, tpr, thr = roc_curve(y, y_pred)

    closest_index = np.argmin(np.abs(tpr - value_target))

    threshold = thr[closest_index]
    false_pos_rate = fpr[closest_index]

    return threshold, false_pos_rate


def find_roc_threshold_fpr(y, y_pred, value_target):
    """
    This function calculates the threshold and true positive rate corresponding to a fixed false positive rate (FPR).

    Parameters:
    y              # Real labels
    y_pred         # Predicted probabilities
    value_target   # Desired false positive rate (FPR) value (between 0 and 1)
    
    Returns:
    threshold       # Threshold value
    true_positive_rate  # Corresponding true positive rate (TPR)
    """
    
    fpr, tpr, thr = roc_curve(y, y_pred)

    # Find the index where FPR is closest to value_target
    closest_index = np.argmin(np.abs(fpr - value_target))

    threshold = thr[closest_index]
    true_pos_rate = tpr[closest_index]

    return threshold, true_pos_rate

def find_roc_threshold_f1(pred_prob, y):
    """
    This function calculates the threshold in the ROC curve that maximizes the F1 score.
    model                    # Trained model
    pred_prob                # Prediction probabilities
    y                        # Target dataset
    
    Returns:
    
    best_threshold           # Threshold value
    best_f1_score            # F1 score value
    """
    
    # Get predicted probabilities for the positive class
    pred_prob = model.predict_proba(X)[:, 1]

    best_threshold = 0.5
    best_f1_score = 0.0
    # Compute the ROC curve (FPR, TPR, thresholds)
    fpr, tpr, thresholds = roc_curve(y, pred_prob)

    for threshold in thresholds:
        # Make predictions based on the threshold
        pred_tmp = np.where(pred_prob >= threshold, 1, 0)
        # Calculate F1 score for this threshold
        score = f1_score(y, pred_tmp)
        
        if score > best_f1_score:
            best_f1_score = score
            best_threshold = threshold

    return best_threshold, best_f1_score

def find_roc_threshold_accuracy(pred_prob, y):
    """
    This function calculates the threshold in the ROC curve that maximizes the accuracy score.
    model                    # Trained model
    pred_prob                # Prediction probabilities
    y                        # Target dataset
    
    Returns:
    
    best_threshold           # Threshold value
    best_acc_score           # Accuracy score value
    """
    
    best_threshold = 0.5
    best_acc_score = 0.0
    # Compute the ROC curve (FPR, TPR, thresholds)
    fpr, tpr, thresholds = roc_curve(y, pred_prob)

    for threshold in thresholds:
        # Make predictions based on the threshold
        pred_tmp = np.where(pred_prob >= threshold, 1, 0)
        # Calculate accuracy score for this threshold
        score = accuracy_score(y, pred_tmp)
        
        if score > best_acc_score:
            best_acc_score = score
            best_threshold = threshold

    return best_threshold, best_acc_score

def partial_auc_score(y_actual, y_pred, tpr_threshold=0.80):
    
    """
    This function calculates the partial AUC score
    y_true: true labels
    y_scores: predictions
    tpr_threshold: true positive rate (recall) threshold above which the auc score will be computed
    """
  
    max_fpr = 1 - tpr_threshold

    # create numpy arrays
    y_actual = np.asarray(y_actual)
    y_pred = np.asarray(y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_actual, y_pred)

    # Find the index where fpr exceeds max_fpr
    stop_index = np.searchsorted(fpr, max_fpr, side='right')

    if stop_index < len(fpr):
        # Interpolate to find the TPR at max_fpr
        fpr_interp_points = [fpr[stop_index - 1], fpr[stop_index]]
        tpr_interp_points = [tpr[stop_index - 1], tpr[stop_index]]
        tpr = np.append(tpr[:stop_index], np.interp(max_fpr, fpr_interp_points, tpr_interp_points))
        fpr = np.append(fpr[:stop_index], max_fpr)
    else:
        tpr = np.append(tpr, 1.0)
        fpr = np.append(fpr, max_fpr)

    # Calculate partial AUC
    partial_auc_value = auc(fpr, tpr)

    return partial_auc_value

def cross_val_partial_auc_score(X, y, model, n_splits):

    """
    This fuction calculates the average partial AUC score across all validation folds.
    X: input vector
    y: label
    model: machine learning model to train and cross-validate
    n_splits: number of k folds
    """

     # Setup cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pauc_scores = []
    cont = 1
    for train_idx, val_idx in skf.split(X, y):

        print(f'Processing fold {cont} of {n_splits}... ', end='', flush=True)
        
        # Create the folds
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
        # Train the model
        model.fit(X_train_fold, y_train_fold)
    
        # Predict on the validation set
        preds = model.predict_proba(X_val_fold)[:,1]
   
        # Calculate partical AUC and store it
        pauc = partial_auc_score(y_val_fold, preds)
        pauc_scores.append(pauc)

        print(f'pAUC: {pauc}', flush=True)
        
        cont += 1
    
    # Return the average
    return np.mean(pauc_scores)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges,
                          figsize=(10,10)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # Confusion matrix
    #cm = confusion_matrix(test_labels, rf_predictions)
    #plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
    #                      title = 'Health Confusion Matrix')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    # Plot the confusion matrix
    plt.figure(figsize = figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    #plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

def plot_class_distribution(y, y_pred, labels):
    """
    This function plots the statistical distrution of the two predicted classes.
    """
    df = pd.DataFrame({'Predicted': list(y_pred), 'Real': list(y)})
    df.Predicted = df.Predicted.astype(float)
    df.Real = df.Real.astype(int)
    sns.kdeplot(data=df, x='Predicted', hue='Real', fill=True, alpha=0.2, linewidth=1.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Values for Each Class')
    plt.legend(labels=labels)
   # plt.show()