"""
Contains functions for training and testing a PyTorch model.
"""

import os
import glob
import logging
import torch
import torchvision
import random
import time
import numpy as np
import pandas as pd
import copy
from datetime import datetime
from typing import Tuple, Dict, Any, List, Union, Optional
from tqdm.auto import tqdm 
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
from torch import GradScaler, autocast
from sklearn.metrics import precision_recall_curve, classification_report, roc_curve, auc
from contextlib import nullcontext
from sklearn.preprocessing import LabelEncoder

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Define the list of valid extensions
    valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

    # Create model save path
    assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
    model_save_path = Path(target_dir) / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving best model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    """Loads a PyTorch model from a target directory.

    Args:
        model: A target PyTorch model to load.
        target_dir: A directory where the model is located.
        model_name: The name of the model to load. Should include
        ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.

    Example usage:
        model = load_model(model=model,
                        target_dir="models",
                        model_name="model.pth")

    Returns:
    The loaded PyTorch model.
    """

    # Create the model directory path
    model_dir_path = Path(target_dir)

    # Define the list of valid extensions
    valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

    # Create model save path
    assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
    model_save_path = Path(target_dir) / model_name

    # Load the model
    print(f"[INFO] Loading model from: {model_path}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model


# Training and prediction engine class
class Engine:

    """
    A class to handle training, evaluation, and predictions for a PyTorch model.

    This class provides functionality to manage the training and evaluation lifecycle
    of a PyTorch model, including saving the best model based on specific criteria.

    Args:
        model (torch.nn.Module, optional): The PyTorch model to handle. Must be instantiated.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(
        self,
        model: torch.nn.Module=None,
        device: str="cuda" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()

        # Initialize self variables
        self.device = device
        self.model = model
        self.save_best_model = False
        self.mode = None
        self.model_best = None
        self.model_epoch = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.model_name = None
        self.model_name_loss = None
        self.model_name_acc = None
        self.model_name_fpr = None
        self.model_name_pauc = None

        # Check if model is provided
        if self.model is None:
            warnings.warn(
                "[WARNING] No model has been introduced. Only limited functionalities "
                "will be allowed: 'sec_to_min_sec', 'calculate_accuracy', "
                "'calculate_fpr_at_recall', 'calculate_pauc_at_recall', "
                "'load', and 'create_writer'."
                "You can later on load the model with 'load'."
            )
        else:
            self.model.to(self.device)
     
        # Create empty results dictionary
        self.results = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_fpr": [],
            "train_pauc": [],
            "train_time [s]": [],
            "test_loss": [],
            "test_acc": [],
            "test_fpr": [],
            "test_pauc": [],
            "test_time [s]": [],
            "lr": [],
            } 

    @staticmethod
    def sec_to_min_sec(seconds):
        """
        Converts seconds to a formatted string in minutes and seconds, fully aligned.

        Args:
            seconds (float): The number of seconds to be converted.
            max_minutes_width (int): The width to align the minutes column.

        Returns:
            str: A formatted string representing the time in minutes and seconds, aligned properly.
        """
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise ValueError("Input must be a non-negative number.")
        
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        # Format aligned with right-justification
        return f"{str(minutes).rjust(3)}m{str(remaining_seconds).zfill(2)}s"

    
    # Calculate accuracy (a classification metric)
    @staticmethod
    def calculate_accuracy(
        y_true,
        y_pred):

        """Calculates accuracy between truth labels and predictions.

        Args:
            y_true (torch.Tensor): Truth labels for predictions.
            y_pred (torch.Tensor): Predictions to be compared to predictions.

        Returns:
            [torch.float]: Accuracy value between y_true and y_pred, e.g. 0.78
        """

        assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
        return torch.eq(y_true, y_pred).sum().item() / len(y_true)

    @staticmethod
    def calculate_fpr_at_recall(
        y_true,
        y_pred_probs,
        recall_threshold):

        """Calculates the False Positive Rate (FPR) at a specified recall threshold.

        Args:
            y_true (torch.Tensor): Ground truth labels.
            y_pred_probs (torch.Tensor): Predicted probabilities.
            recall_threshold (float): The recall threshold at which to calculate the FPR.

        Returns:
            float: The calculated FPR at the specified recall threshold.
        """

        # Check if recall_threhold is a valid number
        if not (0 <= recall_threshold <= 1):
            raise ValueError("recall_threshold must be between 0 and 1.")

        # Convert list to tensor if necessary
        if isinstance(y_pred_probs, list):
            y_pred_probs = torch.cat(y_pred_probs)  
        if isinstance(y_true, list):
            y_true = torch.cat(y_true)

        n_classes = y_pred_probs.shape[1]
        fpr_per_class = []

        for class_idx in range(n_classes):
            # Get true binary labels and predicted probabilities for the class
            y_true_bin = (y_true == class_idx).int().detach().numpy()
            y_scores = y_pred_probs[:, class_idx].detach().numpy()

            # Compute precision-recall curve
            _, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)

            # Find the threshold closest to the desired recall
            idx = np.where(recall >= recall_threshold)[0]
            if len(idx) > 0:
                threshold = thresholds[idx[-1]]
                # Calculate false positive rate
                fp = np.sum((y_scores >= threshold) & (y_true_bin == 0))
                tn = np.sum((y_scores < threshold) & (y_true_bin == 0))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                fpr = 0  # No threshold meets the recall condition

            fpr_per_class.append(fpr)

        return np.mean(fpr_per_class)  # Average FPR across all classes
    
    @staticmethod
    def calculate_pauc_at_recall(
        y_true,
        y_pred_probs,
        recall_threshold=0.80,
        num_classes=101):
        """
        Calculates the Partial AUC for multi-class classification at the given recall threshold.
        Args:
            y_true (list or np.ndarray): Ground truth labels (int or one-hot encoded).
            y_pred_probs (np.ndarray): Predicted probabilities (shape: [batch_size, num_classes]).
            recall_threshold (float): The threshold on recall to calculate partial AUC at.
            num_classes (int): Number of classes in the classification problem.
            
        Returns:
            float: The averaged partial AUC over all classes.
        """
        
        # Ensure the input is in the correct format
        y_true = np.asarray(y_true)
        y_pred_probs = np.asarray(y_pred_probs)

        # Initialize list to store partial AUC values for each class
        partial_auc_values = []

        for class_idx in range(num_classes):
            # Get true binary labels for class_idx and predicted scores
            y_true_bin = (y_true == class_idx).astype(int)
            y_scores_class = y_pred_probs[:, class_idx]

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true_bin, y_scores_class)

            # Find the index where TPR exceeds the recall threshold
            max_fpr = 1 - recall_threshold
            stop_index = np.searchsorted(fpr, max_fpr, side='right')

            # Interpolate to find the TPR at the given max FPR
            if stop_index < len(fpr):
                # Interpolate to find the TPR at max_fpr
                fpr_interp_points = [fpr[stop_index - 1], fpr[stop_index]]
                tpr_interp_points = [tpr[stop_index - 1], tpr[stop_index]]
                tpr = np.append(tpr[:stop_index], np.interp(max_fpr, fpr_interp_points, tpr_interp_points))
                fpr = np.append(fpr[:stop_index], max_fpr)
            else:
                tpr = np.append(tpr, 1.0)
                fpr = np.append(fpr, max_fpr)

            # Calculate partial AUC for the class
            partial_auc_value = auc(fpr, tpr)
            partial_auc_values.append(partial_auc_value)

        # Return the average partial AUC across all classes
        return np.mean(partial_auc_values)


    @staticmethod
    def save(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str):

        """Saves a PyTorch model to a target directory.

        Args:
            model: A target PyTorch model to save.
            target_dir: A directory for saving the model to.
            model_name: A filename for the saved model. Should include
            ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.

        Example usage:
            save(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
        """

        # Create target directory
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True,
                            exist_ok=True)

        # Define the list of valid extensions
        valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

        # Create model save path
        assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
        model_save_path = Path(target_dir) / model_name

        # Save the model state_dict()
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)

    def load(
        self,
        target_dir: str,
        model_name: str,
        return_model: bool=False):
        
        """Loads a PyTorch model from a target directory and optionally returns it.

        Args:
            target_dir: A directory where the model is located.
            model_name: The name of the model to load. Should include
            ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.
            return_model: Whether to return the loaded model (default: False).

        Returns:
            The loaded PyTorch model (if return_model=True).
        """

        # Create the model path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_path = Path(target_dir) / model_name

        # Load the model
        print(f"[INFO] Loading model from: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        
        if return_model:
            return self.model
    
    @staticmethod
    def create_writer(
        experiment_name: str, 
        model_name: str, 
        extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():

        """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

        log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

        Where timestamp is the current date in YYYY-MM-DD format.

        Args:
            experiment_name (str): Name of experiment.
            model_name (str): Name of model.
            extra (str, optional): Anything extra to add to the directory. Defaults to None.

        Returns:
            torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

        Example usage:
            # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
            writer = create_writer(experiment_name="data_10_percent",
                                model_name="effnetb2",
                                extra="5_epochs")
            # The above is the same as:
            writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
        """

        # Get timestamp of current date (all experiments on certain day live in same folder)
        timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

        if extra:
            # Create log directory path
            log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
        else:
            log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
            
        print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
        return SummaryWriter(log_dir=log_dir)
    
    def print_config(
            self,
            batch_size,
            recall_threshold,
            recall_threshold_pauc,
            epochs,
            plot_curves,
            amp,
            enable_clipping,
            accumulation_steps,
            writer):
        
        """
        Prints the configuration of the training process.
        """

        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Epochs: {epochs}")
        print(f"[INFO] Batch size: {batch_size}")
        print(f"[INFO] Accumulation steps: {accumulation_steps}")
        print(f"[INFO] Effective batch size: {batch_size * accumulation_steps}")
        print(f"[INFO] Recall threshold - FPR: {recall_threshold}")
        print(f"[INFO] Recall threshold - pAUC: {recall_threshold_pauc}")
        print(f"[INFO] Plot curves: {plot_curves}")
        print(f"[INFO] Automatic Mixed Precision (AMP): {amp}")
        print(f"[INFO] Enable clipping: {enable_clipping}")
        print(f"[INFO] Enable writer: {writer}")
        print(f"[INFO] Target directory: {self.target_dir}")
        print(f"[INFO] Save model: {self.save_best_model}")
        if self.save_best_model:
          # Extract base name and extension from the model name
            base_name, extension = os.path.splitext(self.model_name)
            
            # Print base name and extension
            print(f"[INFO] Model name base: {base_name}")
            print(f"[INFO] Model name extension: {extension}")
            
            # Iterate over modes and format model name, skipping 'last'
            for mode in self.mode:
                if mode == "last" or mode == "all":
                    # Skip adding 'last' and just use epoch in the filename
                    model_name_with_mode = f"_epoch<int>{extension}"
                else:
                    # For other modes, include mode and epoch in the filename
                    model_name_with_mode = f"_{mode}_epoch<int>{extension}"
                
                # Print the final model save path for each mode
                print(f"[INFO] Save best model based on {mode}: {base_name + model_name_with_mode}")

    def init_train(
        self,
        target_dir: str=None,
        model_name: str=None,
        save_best_model: Union[str, List[str]] = "last",  # Allow both string and list
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        scheduler: torch.optim.lr_scheduler=None,
        batch_size: int=64,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=True,
        accumulation_steps: int=1,
        writer: Optional[SummaryWriter] = None
    ):

        """
        Initializes the training process by setting up the required configurations, parameters, and resources.

        Args:
            target_dir (str, optional): Directory to save the models. Defaults to "models" if not provided.
            model_name (str, optional): Name of the model file to save. Defaults to the class name of the model with ".pth" extension.
            optimizer (torch.optim.Optimizer, optional): The optimizer to minimize the loss function.
            loss_fn (torch.nn.Module, optional): The loss function to minimize during training.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler for the optimizer.
            batch_size (int, optional): Batch size for the training process. Default is 64.
            recall_threshold (float, optional): Recall threshold for fpr calculation. Must be a float between 0.0 and 1.0. Default is 0.95.
            recall_threshold (float, optional): Recall threshold for pAUC calculation. Must be a float between 0.0 and 1.0. Default is 0.95.
            epochs (int, optional): Number of epochs to train. Must be an integer greater than or equal to 1. Default is 30.
            plot_curves (bool, optional): Whether to plot training and validation curves. Default is True.
            amp (bool, optional): Enable automatic mixed precision for faster training. Default is True.
            enable_clipping (bool, optional): Whether to enable gradient clipping. Default is True.
            accumulation_steps (int, optional): Steps for gradient accumulation. Must be an integer greater than or equal to 1. Default is 1.
            save_best_model (Union[str, List[str]]): Criterion mode for saving the model: 
                - "loss": saves the epoch with the lowest validation loss
                - "acc": saves the epoch with the highest validation accuracy
                - "fpr": saves the eopch with the lowsest false positive rate at recall
                - "pauc": saves the epoch with the highest partial area under the curve at recall
                - "last": saves last epoch
                - "all": saves models for all epochs
                - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
                - None: the model will not be saved.
            writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging metrics. Default is False.

        Functionality:
            Validates `recall_threshold`, `accumulation_steps`, and `epochs` parameters with assertions.
            Prints configuration parameters using the `print_config` method.
            Initializes the optimizer, loss function, and scheduler.
            Ensures the target directory for saving models exists, creating it if necessary.
            Sets the model name for saving, defaulting to the model's class name if not provided.
            Initializes structures to track the best-performing model and epoch-specific models:
            
        This method sets up the environment for training, ensuring all necessary resources and parameters are prepared.
        """
      
        # Validate recall_threshold
        if not isinstance(recall_threshold, (int, float)) or not (0.0 <= float(recall_threshold) <= 1.0):
            raise ValueError("[ERROR] recall_threshold must be a float between 0.0 and 1.0.")

        # Validate recall_threshold_pauc
        if not isinstance(recall_threshold_pauc, (int, float)) or not (0.0 <= float(recall_threshold_pauc) <= 1.0):
            raise ValueError("[ERROR] recall_threshold_pauc must be a float between 0.0 and 1.0.")

        # Validate accumulation_steps
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError("[ERROR] accumulation_steps must be an integer greater than or equal to 1.")

        # Validate epochs
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("[ERROR] epochs must be an integer greater than or equal to 1.")

        # Ensure save_best_model is correctly handled
        if save_best_model is None:
            self.save_best_model = False
            mode = []
        elif isinstance(save_best_model, (str, list)):
            self.save_best_model = True
            mode = [save_best_model] if isinstance(save_best_model, str) else save_best_model  # Ensure mode is a list
        else:
            raise ValueError("[ERROR] save_best_model must be None, a string, or a list of strings.")

        # Validate mode only if save_best_model is True
        valid_modes = {"loss", "acc", "fpr", "pauc", "last", "all"}
        if self.save_best_model:
            if not isinstance(mode, list):
                raise ValueError("[ERROR] mode must be a string or a list of strings.")

            for m in mode:
                if m not in valid_modes:
                    raise ValueError(f"[ERROR] Invalid mode value: '{m}'. Must be one of {valid_modes}")

        # Assign the validated mode list
        self.mode = mode

        # Initialize model name path and ensure target directory exists
        self.target_dir = target_dir if target_dir is not None else "models"
        os.makedirs(self.target_dir, exist_ok=True)  
        self.model_name = model_name if model_name is not None else f"model.pth"

        # List of acceptable extensions
        valid_extensions = ['.pth', '.pt', '.pkl', '.h5', '.torch']

        # Check if model_name already has a valid extension, otherwise add the default .pth extension
        if not any(self.model_name.endswith(ext) for ext in valid_extensions):
            self.model_name += '.pth'

        # Print configuration parameters
        self.print_config(
            batch_size=batch_size,
            recall_threshold=recall_threshold,
            recall_threshold_pauc=recall_threshold_pauc,
            epochs=epochs,
            plot_curves=plot_curves,
            amp=amp,
            enable_clipping=enable_clipping,
            accumulation_steps=accumulation_steps,            
            writer=writer
            )
        
        # Initialize optimizer, loss_fn, and scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
    
        # Initialize the best model and model_epoch list based on the specified mode.
        if self.save_best_model:
            if self.mode != "all":
                self.model_best = copy.deepcopy(self.model)                            
                self.model_best.to(self.device)
                self.model_name_loss = self.model_name.replace(".", f"_loss.")
                self.model_name_acc = self.model_name.replace(".", f"_acc.")
                self.model_name_fpr = self.model_name.replace(".", f"_fpr.")
                self.model_name_pauc = self.model_name.replace(".", f"_pauc.")
                self.best_test_loss = float("inf") 
                self.best_test_acc = 0.0
                self.best_test_fpr = float("inf")
                self.best_test_pauc = 0.0
            else:
                self.model_epoch = []
                for k in range(epochs):
                    self.model_epoch.append(copy.deepcopy(self.model))
                    self.model_epoch[k].to(self.device)

    
    def train_step(
        self,
        dataloader: torch.utils.data.DataLoader, 
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        amp: bool=True,
        enable_clipping=True,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
        
        """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
            dataloader: A DataLoader instance for the model to be trained on.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1).
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, train_fpr, train_pauc). For example: (0.1112, 0.8743, 0.01123, 0.15561).
        """

        # Put model in train mode
        self.model.train()
        self.model.to(self.device)

        # Initialize the GradScaler for  Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0    
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)): 
            #, desc=f"Training epoch {epoch_number}..."):

            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.model(X)
                    y_pred = y_pred.contiguous()
                    
                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue

                    # Calculate  and accumulate loss
                    loss = self.loss_fn(y_pred, y)
                
                    # Check for NaN or Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[WARNING] Loss is NaN or Inf at batch {batch}. Skipping batch...")
                        continue

                # Backward pass with scaled gradients
                if debug_mode:
                    # Use anomaly detection
                    with torch.autograd.detect_anomaly():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

                # Gradient clipping
                if enable_clipping:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Check gradients for NaN or Inf values
                if debug_mode:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                print(f"[WARNING] NaN or Inf gradient detected in {name} at batch {batch}.")
                                break
                
                # scaler.step() first unscales the gradients of the optimizer's assigned parameters.
                # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(self.optimizer)
                scaler.update()

                # Optimizer zero grad
                self.optimizer.zero_grad()

            else:
                # Forward pass
                y_pred = self.model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate  and accumulate loss
                loss = self.loss_fn(y_pred, y)

                # Optimizer zero grad
                self.optimizer.zero_grad()

                # Loss backward
                loss.backward()

                # Gradient clipping
                if enable_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                self.optimizer.step()

            # Calculate and accumulate loss and accuracy across all batches
            train_loss += loss.item()
            y_pred_class = y_pred.argmax(dim=1)
            train_acc += self.calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item()/len(y_pred)
            
            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        # Final FPR calculation
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            train_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
            train_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc)
        except Exception as e:
            logging.error(f"[ERROR] Error calculating final FPR at recall: {e}")
            train_fpr = None
            train_pauc = None

        return train_loss, train_acc, train_fpr, train_pauc

    # This train step function includes gradient accumulation (experimental)
    def train_step_v2(
        self,
        dataloader: torch.utils.data.DataLoader, 
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        amp: bool=True,
        enable_clipping=False,
        accumulation_steps: int = 1,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
    
        """Trains a PyTorch model for a single epoch with gradient accumulation.

        Args:
            dataloader: A DataLoader instance for the model to be trained on.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1)
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            accumulation_steps: Number of mini-batches to accumulate gradients before an optimizer step.
                If batch size is 64 and accumulation_steps is 4, gradients are accumulated for 256 mini-batches before an optimizer step.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, train_fpr, train_pauc). For example: (0.1112, 0.8743, 0.01123, 0.15561).
        """

        # Put model in train mode
        self.model.train()
        self.model.to(self.device)

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0    
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.model(X)
                    y_pred = y_pred.contiguous()

                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue
                    
                    # Calculate loss, normalize by accumulation steps
                    loss = self.loss_fn(y_pred, y) / accumulation_steps
                
                    # Check for NaN or Inf in loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"[WARNING] Loss is NaN or Inf at batch {batch}. Skipping...")
                        continue

                # Backward pass with scaled gradients
                if debug_mode:
                    # Use anomaly detection
                    with torch.autograd.detect_anomaly():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

            else:
                # Forward pass
                y_pred = self.model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate loss, normalize by accumulation steps
                loss = self.loss_fn(y_pred, y) / accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient cliping
                if enable_clipping:
                    # Apply clipping if needed
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Perform optimizer step and clear gradients every accumulation_steps
            if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(dataloader):

                if amp:         
                    # Gradient cliping
                    if enable_clipping:
                        # Unscale the gradients before performing any operations on them
                        scaler.unscale_(self.optimizer)
                        # Apply clipping if needed
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Check gradients for NaN or Inf values
                    if debug_mode:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                    print(f"[WARNING] NaN or Inf gradient detected in {name} at batch {batch}")
                                    break

                    # scaler.step() first unscales the gradients of the optimizer's assigned parameters.
                    # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(self.optimizer)
                    scaler.update()

                else:
                    # Optimizer step
                    self.optimizer.step()

                # Optimizer zero grad
                self.optimizer.zero_grad()

            # Accumulate metrics
            train_loss += loss.item() * accumulation_steps  # Scale back to original loss
            y_pred_class = y_pred.argmax(dim=1)
            train_acc += self.calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item() / len(y_pred)

            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        # Final FPR calculation
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            train_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
            train_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc)
        except Exception as e:
            logging.error(f"[ERROR] Error calculating final FPR at recall: {e}")
            train_fpr = None
            train_pauc = None

        return train_loss, train_acc, train_fpr, train_pauc

    def test_step(
        self,
        dataloader: torch.utils.data.DataLoader, 
        recall_threshold: float = 0.95,
        recall_threshold_pauc: float = 0.95,
        amp: bool = True,
        debug_mode: bool = False,
        enable_clipping: bool = False
    ) -> Tuple[float, float, float]:
        
        """Tests a PyTorch model for a single epoch.

        Args:
            dataloader: A DataLoader instance for the model to be tested on.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1).
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            amp: Whether to use Automatic Mixed Precision for inference.
            debug_mode: Enables logging for debugging purposes.
            enable_clipping: Enables NaN/Inf value clipping for test predictions.

        Returns:
            A tuple of test loss, test accuracy, FPR-at-recall, and pAUC-at-recall metrics.
        """

        # Put model in eval mode
        self.model.eval() 
        self.model.to(self.device)

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        all_preds = []
        all_labels = []

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), colour='#FF9E2C'):
                #, desc=f"Validating epoch {epoch_number}..."):
                # Send data to target device
                X, y = X.to(self.device), y.to(self.device)

                # Enable AMP if specified
                with torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext():
                    test_pred = self.model(X)
                    test_pred = test_pred.contiguous()

                    # Check for NaN/Inf in predictions
                    if torch.isnan(test_pred).any() or torch.isinf(test_pred).any():
                        if enable_clipping:
                            print(f"[WARNING] Predictions contain NaN/Inf at batch {batch}. Applying clipping...")
                            test_pred = torch.nan_to_num(
                                test_pred,
                                nan=torch.mean(test_pred).item(),
                                posinf=torch.max(test_pred).item(),
                                neginf=torch.min(test_pred).item()
                            )
                        else:
                            print(f"[WARNING] Predictions contain NaN/Inf at batch {batch}. Skipping batch...")
                            continue

                    # Calculate and accumulate loss
                    loss = self.loss_fn(test_pred, y)
                    test_loss += loss.item()

                    # Debug NaN/Inf loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"[WARNING] Loss is NaN/Inf at batch {batch}. Skipping...")
                        continue

                # Calculate and accumulate accuracy
                test_pred_class = test_pred.argmax(dim=1)
                test_acc += self.calculate_accuracy(y, test_pred_class) #((test_pred_class == y).sum().item()/len(test_pred))

                # Collect outputs for fpr-at-recall calculation
                all_preds.append(torch.softmax(test_pred, dim=1).detach().cpu())
                all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        # Final FPR calculation
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            test_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
            test_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc)
        except Exception as e:
            logging.error(f"[ERROR] Error calculating final FPR at recall: {e}")
            test_fpr = None
            test_pauc = None

        return test_loss, test_acc, test_fpr, test_pauc

    
    def display_results(
        self,
        epoch,
        train_loss,
        train_acc,
        recall_threshold,
        recall_threshold_pauc,
        train_fpr,
        train_pauc,
        train_epoch_time,
        test_loss,
        test_acc,
        test_fpr,
        test_pauc,
        test_epoch_time,
        plot_curves,
        writer
        ):
    
        """
        Displays the training and validation results both numerically and visually.

        Functionality:
        - Outputs key metrics such as training and validation loss, accuracy, and fpr at recall in numerical form.
        - Generates plots that visualize the training process, such as:
        - Loss curves (training vs validation loss over epochs).
        - Accuracy curves (training vs validation accuracy over epochs).
        - FPR at recall curves
        """

        # Define colors
        BLACK = '\033[30m'  # Black
        BLUE = '\033[34m'   # Blue
        ORANGE = '\033[38;5;214m'  # Orange (using extended colors)
        GREEN = '\033[32m'  # Green
        RESET = '\033[39m'  # Reset to default color

        # Retrieve the learning rate
        if self.scheduler is None or isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr = self.optimizer.param_groups[0]['lr']
        else:
            lr = self.scheduler.get_last_lr()[0]
        
        # Print results
        print(
            f"{BLACK}Epoch: {epoch+1} | "
            f"{BLUE}Train: {BLACK}| "
            f"{BLUE}loss: {train_loss:.4f} {BLACK}| "
            f"{BLUE}acc: {train_acc:.4f} {BLACK}| "
            f"{BLUE}fpr: {train_fpr:.4f} {BLACK}| "
            f"{BLUE}pauc: {train_pauc:.4f} {BLACK}| "
            f"{BLUE}time: {self.sec_to_min_sec(train_epoch_time)} {BLACK}| "            
            f"{BLUE}lr: {lr:.10f}"
        )
        print(
            f"{BLACK}Epoch: {epoch+1} | "
            f"{ORANGE}Test:  {BLACK}| "
            f"{ORANGE}loss: {test_loss:.4f} {BLACK}| "
            f"{ORANGE}acc: {test_acc:.4f} {BLACK}| "
            f"{ORANGE}fpr: {test_fpr:.4f} {BLACK}| "
            f"{ORANGE}pauc: {test_pauc:.4f} {BLACK}| "
            f"{ORANGE}time: {self.sec_to_min_sec(test_epoch_time)} {BLACK}| "
            f"{ORANGE}lr: {lr:.10f}"
        )
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)
        self.results["train_loss"].append(train_loss)
        self.results["train_acc"].append(train_acc)
        self.results["test_loss"].append(test_loss)
        self.results["test_acc"].append(test_acc)
        self.results["train_time [s]"].append(train_epoch_time)
        self.results["test_time [s]"].append(test_epoch_time)
        self.results["lr"].append(lr)
        self.results["train_fpr"].append(train_fpr)
        self.results["test_fpr"].append(test_fpr)
        self.results["train_pauc"].append(train_pauc)
        self.results["test_pauc"].append(test_pauc)
        
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss", 
                tag_scalar_dict={"train_loss": train_loss,
                                 "test_loss": test_loss},
                global_step=epoch)
            writer.add_scalars(
                main_tag="Accuracy", 
                tag_scalar_dict={"train_acc": train_acc,
                                 "test_acc": test_acc},
                global_step=epoch)
            writer.add_scalars(
                main_tag=f"FPR at {recall_threshold * 100}% recall", 
                tag_scalar_dict={"train_fpr": train_fpr,
                                    "test_fpr": test_fpr}, 
                global_step=epoch)
            writer.add_scalars(
                main_tag=f"pAUC above {recall_threshold_pauc * 100}% recall", 
                tag_scalar_dict={"train_pauc": train_pauc,
                                    "test_pauc": test_pauc}, 
                global_step=epoch)
        else:
            pass

        # Plots training and test loss, accuracy, and fpr-at-recall curves.
        if plot_curves:
        
            n_plots = 4
            plt.figure(figsize=(20, 6))
            range_epochs = range(1, len(self.results["train_loss"])+1)

            # Plot loss
            plt.subplot(1, n_plots, 1)
            plt.plot(range_epochs, self.results["train_loss"], label="train_loss")
            plt.plot(range_epochs, self.results["test_loss"], label="test_loss")
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot accuracy
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, self.results["train_acc"], label="train_accuracy")
            plt.plot(range_epochs, self.results["test_acc"], label="test_accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            # Plot FPR at recall
            plt.subplot(1, n_plots, 3)
            plt.plot(range_epochs, self.results["train_fpr"], label="train_fpr_at_recall")
            plt.plot(range_epochs, self.results["test_fpr"], label="test_fpr_at_recall")
            plt.title(f"FPR at {recall_threshold * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot pAUC at recall
            plt.subplot(1, n_plots, 4)
            plt.plot(range_epochs, self.results["train_pauc"], label="train_pauc_at_recall")
            plt.plot(range_epochs, self.results["test_pauc"], label="test_pauc_at_recall")
            plt.title(f"pAUC above {recall_threshold_pauc * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            plt.show()

    # Scheduler step after the optimizer
    def scheduler_step(
        self,
        test_loss: float=None,
        test_acc: float=None,
        ):

        """
        Performs a scheduler step after the optimizer step.

        Parameters:
        - scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        - test_loss (float, optional): Test loss value, required for ReduceLROnPlateau with 'min' mode.
        - test_acc (float, optional): Test accuracy value, required for ReduceLROnPlateau with 'max' mode.
        """
            
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Check whether scheduler is configured for "min" or "max"
                if self.scheduler.mode == "min" and test_loss is not None:
                    self.scheduler.step(test_loss)  # Minimize test_loss
                elif self.scheduler.mode == "max" and test_acc is not None:
                    self.scheduler.step(test_acc)  # Maximize test_accuracy
                else:
                    raise ValueError(
                        "The scheduler requires either `test_loss` or `test_acc` "
                        "depending on its mode ('min' or 'max')."
                        )
            else:
                self.scheduler.step()  # For other schedulers
    
    # Updates and saves the best model and model_epoch list based on the specified mode.
    def update_model(
        self,
        test_loss: float = None,
        test_acc: float = None,
        test_fpr: float = None,
        test_pauc: float = None,
        epoch: int = None,
    ) -> pd.DataFrame:

        """
        Updates and saves the best model and model_epoch list based on the specified mode(s), as well as the last-epoch model.

        Parameters:
        - test_loss (float, optional): Test loss for the current epoch (used in "loss" mode).
        - test_acc (float, optional): Test accuracy for the current epoch (used in "acc" mode).
        - test_fpr (float, optional): Test false positive rate at the specified recall (used in "fpr" mode).
        - test_pauc (float, optional): Test pAUC at the specified recall (used in "pauc" mode).
        - epoch (int, optional): Current epoch index, used for naming models when saving all epochs in "all" mode.

        Functionality:
        - Saves the last-epoch model.
        - Saves the logs (self.results).
        - Saves the best-performing model during training based on the specified evaluation mode.
        - If mode is "all", saves the model for every epoch.
        - Updates `self.model_best` and `self.model_epoch` accordingly.

        Returns:
            A dataframe of training and testing metrics for each epoch.
        """

        if isinstance(self.mode, str):
            self.mode = [self.mode]  # Ensure self.mode is always a list

        if epoch is None:
            raise ValueError("[ERROR] epoch must be provided when mode includes 'all' or 'last'.")

        def remove_previous_best(model_name_pattern):
            """Removes previously saved best model files."""
            file_to_remove = glob.glob(os.path.join(self.target_dir, model_name_pattern.replace(".", "_epoch*.")))
            for f in file_to_remove:
                os.remove(f)

        def save_model(model_name):
            """Helper function to save the model."""
            self.save(model=self.model, target_dir=self.target_dir, model_name=model_name.replace(".", f"_epoch{epoch+1}."))

        if self.save_best_model:
            for mode in self.mode:
                if mode == "loss":
                    if test_loss is None:
                        raise ValueError("[ERROR] test_loss must be provided when mode is 'loss'.")
                    if test_loss < self.best_test_loss:
                        remove_previous_best(self.model_name_loss)
                        self.best_test_loss = test_loss
                        save_model(self.model_name_loss)

                elif mode == "acc":
                    if test_acc is None:
                        raise ValueError("[ERROR] test_acc must be provided when mode is 'acc'.")
                    if test_acc > self.best_test_acc:
                        remove_previous_best(self.model_name_acc)
                        self.best_test_acc = test_acc
                        save_model(self.model_name_acc)

                elif mode == "fpr":
                    if test_fpr is None:
                        raise ValueError("[ERROR] test_fpr must be provided when mode is 'fpr'.")
                    if test_fpr < self.best_test_fpr:
                        remove_previous_best(self.model_name_fpr)
                        self.best_test_fpr = test_fpr
                        save_model(self.model_name_fpr)

                elif mode == "pauc":
                    if test_pauc is None:
                        raise ValueError("[ERROR] test_pauc must be provided when mode is 'pauc'.")
                    if test_pauc > self.best_test_pauc:
                        remove_previous_best(self.model_name_pauc)
                        self.best_test_pauc = test_pauc
                        save_model(self.model_name_pauc)

                elif mode == "last":
                    remove_previous_best(self.model_name)
                    save_model(self.model_name)

                elif mode == "all":
                    save_model(self.model_name)
                    if isinstance(self.model_epoch, list) and epoch < len(self.model_epoch):
                        self.model_epoch[epoch].load_state_dict(self.model.state_dict())

        # Save results to CSV
        name, _ = self.model_name.rsplit('.', 1)
        csv_file_name = f"{name}.csv"
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(os.path.join(self.target_dir, csv_file_name), index=False)

        return df_results

    
    def finish_train(
        self,
        train_time: float=None,
        writer: SummaryWriter=False
        ):

        """
        Finalizes the training process by closing writer and showing the elapsed time.
        
        Args:
            train_time: Elapsed time.
            writer: A SummaryWriter() instance to log model results to.
        """

        # Close the writer
        writer.close() if writer else None

        # Print elapsed time
        print(f"[INFO] Training finished! Elapsed time: {self.sec_to_min_sec(train_time)}")
            
    # Trains and tests a Pytorch model
    def train(
        self,
        target_dir: str=None,
        model_name: str=None,
        save_best_model: Union[str, List[str]] = "last",  # Allow both string and list
        train_dataloader: torch.utils.data.DataLoader=None, 
        test_dataloader: torch.utils.data.DataLoader=None, 
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        scheduler: torch.optim.lr_scheduler=None,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=True,
        accumulation_steps: int=1,
        debug_mode: bool=False,
        writer=False, #: SummaryWriter=False,
        ) -> pd.DataFrame:

        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Stores metrics to specified writer log_dir if present.

        Args:
            target_dir: A directory for saving the model to.
            model_name: A filename for the saved model. Should include
            ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.
            save_best_model (Union[str, List[str]]): Criterion mode for saving the model: 
            - "loss" (validation loss)
            - "acc" (validation accuracy)
            - "fpr" (false positive rate at recall)
            - "pauc" (partial area under the curve at recall)
            - "last" (last epoch)
            - "all" (save models for all epochs)
            - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
            train_dataloader: A DataLoader instance for the model to be trained on.
            test_dataloader: A DataLoader instance for the model to be tested on.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            loss_fn: A PyTorch loss function to calculate loss on both datasets.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1). 
            recall_threshold_pauc: The recall threshold at which to calculate the pAUC score (between 0 and 1).
            scheduler: A PyTorch learning rate scheduler to adjust the learning rate during training.
            epochs: An integer indicating how many epochs to train for.        
            plot: A boolean indicating whether to plot the training and testing curves.
            amp: A boolean indicating whether to use Automatic Mixed Precision (AMP) during training.
            enable_clipping: enables clipping on gradients and model outputs.
            accumulation_steps: An integer indicating how many mini-batches to accumulate gradients before an optimizer step. Default = 1: no accumulation.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.
            writer: A SummaryWriter() instance to log model results to.

        Returns:
            A dataframe of training and testing loss, training and testing accuracy metrics,
            and training testing fpr at recall. Each metric has a value in a list for 
            each epoch.
            In the form: {
                epoch: [...],
                train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...],
                train_time: [...],
                test_time: [...],
                lr: [...],
                train_fpr: [...],
                test_fpr: [...]
                train_pauc: [...],
                test_pauc: [...],
                } 
            For example if training for epochs=2: 
                {
                epoch: [ 1, 2].
                train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973],
                train_time: [1.1234, 1.5678],
                test_time: [0.4567, 0.7890],
                lr: [0.001, 0.0005],
                train_fpr: [0.1234, 0.2345],
                test_fpr: [0.3456, 0.4567]
                train_pauc: [0.1254, 0.3445],
                test_pauc: [0.3154, 0.4817]
                } 
        """

        # Starting training time
        train_start_time = time.time()

        # Initialize training process
        self.init_train(
            target_dir=target_dir,
            model_name=model_name,
            save_best_model=save_best_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            batch_size=train_dataloader.batch_size,
            recall_threshold=recall_threshold,
            recall_threshold_pauc=recall_threshold_pauc,
            epochs=epochs, 
            plot_curves=plot_curves,
            amp=amp,
            enable_clipping=enable_clipping,
            accumulation_steps=accumulation_steps,
            writer=writer
            )

        # Loop through training and testing steps for a number of epochs
        for epoch in range(epochs):

            # Perform training step and time it
            print(f"Training epoch {epoch+1}...")
            train_epoch_start_time = time.time()
            train_loss, train_acc, train_fpr, train_pauc = self.train_step_v2(
                dataloader=train_dataloader,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                amp=amp,
                enable_clipping=enable_clipping,
                accumulation_steps=accumulation_steps,
                debug_mode=debug_mode
                )
            train_epoch_time = time.time() - train_epoch_start_time

            # Perform test step and time it
            print(f"Validating epoch {epoch+1}...")
            test_epoch_start_time = time.time()
            test_loss, test_acc, test_fpr, test_pauc = self.test_step(
                dataloader=test_dataloader,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                amp=amp,
                enable_clipping=enable_clipping,
                debug_mode=debug_mode
                )
            test_epoch_time = time.time() - test_epoch_start_time

            clear_output(wait=True)

            # Show results
            self.display_results(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                train_fpr=train_fpr,
                train_pauc=train_pauc,
                train_epoch_time=train_epoch_time,
                test_loss=test_loss,
                test_acc=test_acc,
                test_fpr=test_fpr,
                test_pauc=test_pauc,
                test_epoch_time=test_epoch_time,
                plot_curves=plot_curves,
                writer=writer
            )

            # Scheduler step after the optimizer
            self.scheduler_step(
                test_loss=test_loss,
                test_acc=test_acc
            )

            # Update and save the best model, model_epoch list based on the specified mode, and the actual-epoch model.
            df_results = self.update_model(
                test_loss=test_loss,
                test_acc=test_acc,
                test_fpr=test_fpr,
                test_pauc=test_pauc,
                epoch=epoch
                )

        # Finish training process
        train_time = time.time() - train_start_time
        self.finish_train(train_time, writer)

        return df_results

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_state: str="default",
        output_type: str="softmax",        
        ) -> torch.Tensor:

        """
        Predicts classes for a given dataset using a trained model.

        Args:
            model_state: specifies the model to use for making predictions. Default: "default", the last stored model.
            dataloader (torch.utils.data.DataLoader): The dataset to predict on.
            output_type (str): The type of output to return. Either "softmax", "logits", or "argmax".            

        Returns:
            (list): All of the predicted class labels represented by prediction probabilities (softmax)
        """
 
        # Check model to use
        valid_models = {"default", "best"}
        assert model_state in valid_models or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."

        if model_state == "default":
            model = self.model
        elif model_state == "best":
            if self.model_best is None:
                print(f"[INFO] Model best not found, using default model for prediction.")
                model = self.model
            else:
                model = self.model_best
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"[INFO] Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"[INFO] Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]            


        # Check output_max
        valid_output_types = {"softmax", "argmax", "logits"}
        assert output_type in valid_output_types, f"Invalid output_max value: {output_type}. Must be one of {valid_output_types}"

        y_preds = []
        model.eval()
        model.to(self.device)
        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Making predictions"):

                # Send data and targets to target device
                X, y = X.to(self.device), y.to(self.device)
                
                # Do the forward pass
                y_logit = model(X)

                if output_type == "softmax":
                    y_pred = torch.softmax(y_logit, dim=1)
                elif output_type == "argmax":
                    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                else:
                    y_pred = y_logit

                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())

        # Concatenate list of predictions into a tensor
        return torch.cat(y_preds)

    def predict_and_store(
        self,
        test_dir: str, 
        transform: torchvision.transforms, 
        class_names: List[str], 
        model_state: str="default",
        sample_fraction: float=1.0,
        seed=42,        
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        
        """
        Predicts classes for a given dataset using a trained model and stores the per-sample results in dictionaries.

        Args:
            model_state: specifies the model to use for making predictions. "last" (default), "best", an integer
            test_dir (str): The directory containing the test images.
            transform (torchvision.transforms): The transformation to apply to the test images.
            class_names (list): A list of class names.
            sample_fraction (float, optional): The fraction of samples to predict. Defaults to 1.0.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            A list of dictionaries with sample predictions, sample names, prediction probabilities, prediction times, actual labels and prediction classes
            A classification report as a dictionary from sckit-learng.metrics
        """

        # Check model to use
        valid_models = {"default", "best"}
        assert model_state in valid_models or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."

        if model_state == "default":
            model = self.model
        elif model_state == "best":
            if self.model_best is None:
                print(f"[INFO] Model best not found, using default model for prediction.")
                model = self.model
            else:
                model = self.model_best
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"[INFO] Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"[INFO] Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]

        # Create a list of test images and checkout existence
        print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
        paths = list(Path(test_dir).glob("*/*.jpg"))
        assert len(list(paths)) > 0, f"No files ending with '.jpg' found in this directory: {test_dir}"

        # Number of random images to extract
        num_samples = len(paths)
        num_random_images = int(sample_fraction * num_samples)

        # Ensure the number of images to extract is less than or equal to the total number of images
        assert num_random_images <= len(paths), f"Number of images to extract exceeds total images in directory: {len(paths)}"

        # Randomly select a subset of file paths
        torch.manual_seed(seed)
        paths = random.sample(paths, num_random_images)

        # Store predictions and ground-truth labels
        y_true = []
        y_pred = []

        # Create an empty list to store prediction dictionaires
        pred_list = []
        
        # Loop through target paths
        for path in tqdm(paths, total=num_samples):
            
            # Create empty dictionary to store prediction information for each sample
            pred_dict = {}

            # Get the sample path and ground truth class name
            pred_dict["image_path"] = path
            class_name = path.parent.stem
            pred_dict["class_name"] = class_name
            
            # Start the prediction timer
            start_time = timer()
            
            # Open image path
            img = Image.open(path)
            
            # Transform the image, add batch dimension and put image on target device
            transformed_image = transform(img).unsqueeze(0).to(self.device) 
            
            # Prepare model for inference by sending it to target device and turning on eval() mode
            model.to(self.device)
            model.eval()
            
            # Get prediction probability, predicition label and prediction class
            with torch.inference_mode():
                pred_logit = model(transformed_image) # perform inference on target sample 
                #pred_logit = pred_logit.contiguous()
                pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
                pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
                pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

                # Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
                pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
                pred_dict["pred_class"] = pred_class
                
                # End the timer and calculate time per pred
                end_time = timer()
                pred_dict["time_for_pred"] = round(end_time-start_time, 4)

            # Does the pred match the true label?
            pred_dict["correct"] = class_name == pred_class

            # Add the dictionary to the list of preds
            pred_list.append(pred_dict)

            # Append true and predicted label indexes
            y_true.append(class_names.index(class_name))
            y_pred.append(pred_label.cpu().item())

        # Ensure the labels match the class indices
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)
        labels = label_encoder.transform(class_names)

        # Generate the classification report
        classification_report_dict = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=class_names,
            labels=labels,
            output_dict=True
            )

        # Return list of prediction dictionaries
        return pred_list, classification_report_dict