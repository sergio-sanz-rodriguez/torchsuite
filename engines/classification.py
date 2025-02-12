"""
Contains classes for training and testing a PyTorch model.  
Currently, the functionality is limited to classification tasks.  
Support for other deep learning tasks, such as object segmentation, will be added in the future.
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
import warnings
from datetime import datetime
from typing import Tuple, Dict, Any, List, Union, Optional
from tqdm.auto import tqdm 
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_recall_curve, classification_report, roc_curve, auc
from contextlib import nullcontext
from sklearn.preprocessing import LabelEncoder


# Common utility class
class Common:

    """A class containing utility functions for classification tasks."""

    colors = {
        "BLACK":  '\033[30m',
        "BLUE": '\033[34m',
        "ORANGE": '\033[38;5;214m',
        "GREEN": '\033[32m',
        "RED": '\033[31m',
        "RESET": '\033[39m'
    }

    info =    f"{colors['GREEN']}[INFO]{colors['BLACK']}"
    warning = f"{colors['ORANGE']}[WARNING]{colors['BLACK']}"
    error =   f"{colors['RED']}[ERROR]{colors['BLACK']}"
    
    @staticmethod
    def sec_to_min_sec(seconds):
        """Converts seconds to a formatted string in minutes and seconds."""
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise ValueError(f"{Common.error} Input must be a non-negative number.")
        
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        return f"{str(minutes).rjust(3)}m{str(remaining_seconds).zfill(2)}s"

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """Calculates accuracy between truth labels and predictions."""
        assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
        return torch.eq(y_true, y_pred).sum().item() / len(y_true)

    @staticmethod
    def calculate_fpr_at_recall(y_true, y_pred_probs, recall_threshold):
        """Calculates the False Positive Rate (FPR) at a specified recall threshold."""
        if not (0 <= recall_threshold <= 1):
            raise ValueError(f"{Common.error} 'recall_threshold' must be between 0 and 1.")

        if isinstance(y_pred_probs, list):
            y_pred_probs = torch.cat(y_pred_probs)
        if isinstance(y_true, list):
            y_true = torch.cat(y_true)

        n_classes = y_pred_probs.shape[1]
        fpr_per_class = []

        for class_idx in range(n_classes):
            y_true_bin = (y_true == class_idx).int().detach().numpy()
            y_scores = y_pred_probs[:, class_idx].detach().numpy()

            _, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)

            idx = np.where(recall >= recall_threshold)[0]
            if len(idx) > 0:
                threshold = thresholds[idx[-1]]
                fp = np.sum((y_scores >= threshold) & (y_true_bin == 0))
                tn = np.sum((y_scores < threshold) & (y_true_bin == 0))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                fpr = 0  

            fpr_per_class.append(fpr)

        return np.mean(fpr_per_class)

    @staticmethod
    def calculate_pauc_at_recall(y_true, y_pred_probs, recall_threshold=0.80, num_classes=101):
        """Calculates the Partial AUC for multi-class classification at the given recall threshold."""
        y_true = np.asarray(y_true)
        y_pred_probs = np.asarray(y_pred_probs)

        partial_auc_values = []

        for class_idx in range(num_classes):
            y_true_bin = (y_true == class_idx).astype(int)
            y_scores_class = y_pred_probs[:, class_idx]

            fpr, tpr, _ = roc_curve(y_true_bin, y_scores_class)

            max_fpr = 1 - recall_threshold
            stop_index = np.searchsorted(fpr, max_fpr, side='right')

            if stop_index < len(fpr):
                fpr_interp_points = [fpr[stop_index - 1], fpr[stop_index]]
                tpr_interp_points = [tpr[stop_index - 1], tpr[stop_index]]
                tpr = np.append(tpr[:stop_index], np.interp(max_fpr, fpr_interp_points, tpr_interp_points))
                fpr = np.append(fpr[:stop_index], max_fpr)
            else:
                tpr = np.append(tpr, 1.0)
                fpr = np.append(fpr, max_fpr)

            partial_auc_value = auc(fpr, tpr)
            partial_auc_values.append(partial_auc_value)

        return np.mean(partial_auc_values)

    @staticmethod
    def save_model(model: torch.nn.Module, target_dir: str, model_name: str):

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
        print(f"{Common.info} Saving best model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)

    @staticmethod
    def load_model(model: torch.nn.Module, target_dir: str, model_name: str):
        
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

        # Define the list of valid extensions
        valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

        # Create model save path
        assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
        model_save_path = Path(target_dir) / model_name

        # Load the model
        print(f"{Common.info} Loading model from: {model_save_path}")
        
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        
        return model
    
    @staticmethod
    def get_predictions(output):
        if isinstance(output, torch.Tensor):
            return output.contiguous()
        elif hasattr(output, "logits"):
            return output.logits.contiguous()
        else:
            raise TypeError(f"Unexpected model output type: {type(output)}")


# Training and prediction engine class
class ClassificationEngine:

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
        self.keep_best_models_in_memory = False
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
        self.get_predictions = Common.get_predictions
     
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

        # Check if model is provided
        if self.model is None:
            raise ValueError(f"{Common.error} Instantiate the engine by passing a PyTorch model to handle.")
        else:
            self.model.to(self.device)

    def save(
        self,
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
        print(f"{Common.info} Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)

    def load(
        self,
        target_dir: str,
        model_name: str):
        
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
        print(f"{Common.info} Loading model from: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        
        return self
    
    
    def create_writer(
        self,
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
            
        print(f"{Common.info} Created SummaryWriter, saving to: {log_dir}...")
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
            debug_mode,
            writer):
        
        """
        Prints the configuration of the training process.
        """

        print(f"{Common.info} Device: {self.device}")
        print(f"{Common.info} Epochs: {epochs}")
        print(f"{Common.info} Batch size: {batch_size}")
        print(f"{Common.info} Accumulation steps: {accumulation_steps}")
        print(f"{Common.info} Effective batch size: {batch_size * accumulation_steps}")
        print(f"{Common.info} Recall threshold - fpr: {recall_threshold}")
        print(f"{Common.info} Recall threshold - pauc: {recall_threshold_pauc}")
        print(f"{Common.info} Apply validation: {self.apply_validation}")
        print(f"{Common.info} Plot curves: {plot_curves}")
        print(f"{Common.info} Automatic Mixed Precision (AMP): {amp}")
        print(f"{Common.info} Enable clipping: {enable_clipping}")
        print(f"{Common.info} Debug mode: {debug_mode}")
        print(f"{Common.info} Enable writer: {writer}")
        print(f"{Common.info} Save model: {self.save_best_model}")
        print(f"{Common.info} Target directory: {self.target_dir}")        
        if self.save_best_model:
          # Extract base name and extension from the model name
            base_name, extension = os.path.splitext(self.model_name)
            
            # Print base name and extension
            print(f"{Common.info} Model name base: {base_name}")
            print(f"{Common.info} Model name extension: {extension}")
            
            # Iterate over modes and format model name, skipping 'last'
            for mode in self.mode:
                if mode == "last" or mode == "all":
                    # Skip adding 'last' and just use epoch in the filename
                    model_name_with_mode = f"_epoch<int>{extension}"
                else:
                    # For other modes, include mode and epoch in the filename
                    model_name_with_mode = f"_{mode}_epoch<int>{extension}"
                
                # Print the final model save path for each mode
                print(f"{Common.info} Save best model - {mode}: {base_name + model_name_with_mode}")
        if self.keep_best_models_in_memory:
            print(f"{Common.warning} Keeping best models in memory: {self.keep_best_models_in_memory} - it may slow down the training process.")
        else:
            print(f"{Common.info} Keeping best models in memory: {self.keep_best_models_in_memory}")

    def init_train(
        self,
        target_dir: str=None,
        model_name: str=None,
        apply_validation: bool=True,
        save_best_model: Union[str, List[str]] = "last",  # Allow both string and list
        keep_best_models_in_memory: bool=False,
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        scheduler: torch.optim.lr_scheduler=None,
        batch_size: int=64,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=False,
        accumulation_steps: int=1,
        debug_mode: bool=False,
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
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.
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

        # Validate keep_best_models_in_memory
        if not isinstance(keep_best_models_in_memory, (bool)):
            raise ValueError(f"{Common.error}'keep_best_models_in_memory' must be True or False.")
        else:
            self.keep_best_models_in_memory = keep_best_models_in_memory

        # Validate apply_validation
        if not isinstance(apply_validation, (bool)):
            raise ValueError(f"{Common.error}'apply_validation' must be True or False.")
        else:
            self.apply_validation = apply_validation
      
        # Validate recall_threshold
        if not isinstance(recall_threshold, (int, float)) or not (0.0 <= float(recall_threshold) <= 1.0):
            raise ValueError(f"{Common.error}'recall_threshold' must be a float between 0.0 and 1.0.")

        # Validate recall_threshold_pauc
        if not isinstance(recall_threshold_pauc, (int, float)) or not (0.0 <= float(recall_threshold_pauc) <= 1.0):
            raise ValueError(f"{Common.error}'recall_threshold_pauc' must be a float between 0.0 and 1.0.")

        # Validate accumulation_steps
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError(f"{Common.error}'accumulation_steps' must be an integer greater than or equal to 1.")

        # Validate epochs
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"{Common.error}'epochs' must be an integer greater than or equal to 1.")

        # Ensure save_best_model is correctly handled
        if save_best_model is None:
            self.save_best_model = False
            mode = []
        elif isinstance(save_best_model, (str, list)):
            self.save_best_model = True
            mode = [save_best_model] if isinstance(save_best_model, str) else save_best_model  # Ensure mode is a list
        else:
            raise ValueError(f"{Common.error}'save_best_model' must be None, a string, or a list of strings.")

        # Validate mode only if save_best_model is True
        valid_modes = {"loss", "acc", "fpr", "pauc", "last", "all"}
        if self.save_best_model:
            if not isinstance(mode, list):
                raise ValueError(f"{Common.error}'mode' must be a string or a list of strings.")

            for m in mode:
                if m not in valid_modes:
                    raise ValueError(f"{Common.error}Invalid mode value: '{m}'. Must be one of {valid_modes}")

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
            debug_mode=debug_mode,
            writer=writer
            )
        
        # Initialize optimizer, loss_fn, and scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
    
        # Initialize the best model and model_epoch list based on the specified mode.
        if self.save_best_model:
            if "loss" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_loss = copy.deepcopy(self.model)                            
                    self.model_loss.to(self.device)
                self.model_name_loss = self.model_name.replace(".", f"_loss.")
            if "acc" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_acc = copy.deepcopy(self.model)                            
                    self.model_acc.to(self.device)
                self.model_name_acc = self.model_name.replace(".", f"_acc.")
            if "fpr" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_fpr = copy.deepcopy(self.model)                            
                    self.model_fpr.to(self.device)
                self.model_name_fpr = self.model_name.replace(".", f"_fpr.")
            if "pauc" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_pauc = copy.deepcopy(self.model)                            
                    self.model_pauc.to(self.device)
                self.model_name_pauc = self.model_name.replace(".", f"_pauc.")
            if "all" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_epoch = []
                    for k in range(epochs):
                        self.model_epoch.append(copy.deepcopy(self.model))
                        self.model_epoch[k].to(self.device)
            self.best_test_loss = float("inf") 
            self.best_test_acc = 0.0
            self.best_test_fpr = float("inf")
            self.best_test_pauc = 0.0
    
    def train_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_classes: int=2,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epoch_number: int = 1,
        amp: bool=True,
        enable_clipping=False,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
        
        """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
            dataloader: A DataLoader instance for the model to be trained on.
            num_classes: Number of classes.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1).
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            epoch_number: Epoch number.
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, train_fpr, train_pauc). For example: (0.1112, 0.8743, 0.01123, 0.15561).
        """

        print(f"{Common.info} Training epoch {epoch_number+1}...")

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
                            print(f"{Common.warning} y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            print(f"{Common.warning} y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue

                    # Calculate  and accumulate loss
                    loss = self.loss_fn(y_pred, y)
                
                    # Check for NaN or Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"{Common.warning} Loss is NaN or Inf at batch {batch}. Skipping batch...")
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
                                print(f"{Common.warning} NaN or Inf gradient detected in {name} at batch {batch}.")
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
            train_acc += Common.calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item()/len(y_pred)
            
            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        # Final FPR calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        try:    
            train_fpr = Common.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
        except Exception as e:
            logging.error(f"{Common.warning} Innacurate calculation of final FPR at recall: {e}")
            train_fpr = 1.0
        try:    
            train_pauc = Common.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
        except Exception as e:
            logging.error(f"{Common.warning} Innacurate calculation of final pAUC at recall: {e}")
            train_pauc = 0.0

        return train_loss, train_acc, train_fpr, train_pauc

    # This train step function includes gradient accumulation (experimental)
    def train_step_v2(
        self,
        dataloader: torch.utils.data.DataLoader, 
        num_classes: int=2,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epoch_number: int = 1,
        amp: bool=True,
        enable_clipping=False,
        accumulation_steps: int = 1,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
    
        """Trains a PyTorch model for a single epoch with gradient accumulation.

        Args:
            dataloader: A DataLoader instance for the model to be trained on.
            num_classes: Number of classes.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1)
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            epoch_number: Epoch number.
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            accumulation_steps: Number of mini-batches to accumulate gradients before an optimizer step.
                If batch size is 64 and accumulation_steps is 4, gradients are accumulated for 256 mini-batches before an optimizer step.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, train_fpr, train_pauc). For example: (0.1112, 0.8743, 0.01123, 0.15561).
        """

        print(f"{Common.info} Training epoch {epoch_number+1}...")

        # Put model in train mode
        self.model.train()
        self.model.to(self.device)

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        len_dataloader = len(dataloader)
        train_loss, train_acc = 0, 0    
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len_dataloader):
            
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.get_predictions(self.model(X))
                    
                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            print(f"{Common.warning} y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            print(f"{Common.warning} y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue
                    
                    # Calculate loss, normalize by accumulation steps
                    loss = self.loss_fn(y_pred, y) / accumulation_steps
                
                    # Check for NaN or Inf in loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"{Common.warning} Loss is NaN or Inf at batch {batch}. Skipping...")
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
                y_pred = self.get_predictions(self.model(X))
                
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
                                    print(f"{Common.warning} NaN or Inf gradient detected in {name} at batch {batch}")
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
            y_pred = y_pred.float() # Convert to float for stability
            y_pred_class = y_pred.argmax(dim=1)
            train_acc += Common.calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item() / len(y_pred)

            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch
        train_loss /= len_dataloader
        train_acc /= len_dataloader

        # Final FPR calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        try:    
            train_fpr = Common.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
        except Exception as e:
            logging.error(f"{Common.warning} Innacurate calculation of final FPR at recall: {e}")
            train_fpr = 1.0
        try:    
            train_pauc = Common.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
        except Exception as e:
            logging.error(f"{Common.warning} Innacurate calculation of final pAUC at recall: {e}")
            train_pauc = 0.0

        return train_loss, train_acc, train_fpr, train_pauc

    def test_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_classes: int=2,
        recall_threshold: float = 0.95,
        recall_threshold_pauc: float = 0.95,
        epoch_number: int = 1,
        amp: bool = True,
        debug_mode: bool = False,
        enable_clipping: bool = False
    ) -> Tuple[float, float, float]:
        
        """Tests a PyTorch model for a single epoch.

        Args:
            dataloader: A DataLoader instance for the model to be tested on.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1).
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            epoch_number: Epoch number.
            amp: Whether to use Automatic Mixed Precision for inference.
            debug_mode: Enables logging for debugging purposes.
            enable_clipping: Enables NaN/Inf value clipping for test predictions.

        Returns:
            A tuple of test loss, test accuracy, FPR-at-recall, and pAUC-at-recall metrics.
        """

        # Execute the test step is apply_validation is enabled
        if self.apply_validation:

            print(f"{Common.info} Validating epoch {epoch_number+1}...")

            # Put model in eval mode
            self.model.eval() 
            self.model.to(self.device)

            # Setup test loss and test accuracy values
            len_dataloader = len(dataloader)
            test_loss, test_acc = 0, 0
            all_preds = []
            all_labels = []

            # Turn on inference context manager
            with torch.inference_mode():
                # Loop through DataLoader batches
                for batch, (X, y) in tqdm(enumerate(dataloader), total=len_dataloader, colour='#FF9E2C'):
                    #, desc=f"Validating epoch {epoch_number}..."):
                    # Send data to target device
                    X, y = X.to(self.device), y.to(self.device)

                    # Enable AMP if specified
                    with torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext():
                        test_pred = self.get_predictions(self.model(X))

                        # Check for NaN/Inf in predictions
                        if torch.isnan(test_pred).any() or torch.isinf(test_pred).any():
                            if enable_clipping:
                                print(f"{Common.warning} Predictions contain NaN/Inf at batch {batch}. Applying clipping...")
                                test_pred = torch.nan_to_num(
                                    test_pred,
                                    nan=torch.mean(test_pred).item(),
                                    posinf=torch.max(test_pred).item(),
                                    neginf=torch.min(test_pred).item()
                                )
                            else:
                                print(f"{Common.warning} Predictions contain NaN/Inf at batch {batch}. Skipping batch...")
                                continue

                        # Calculate and accumulate loss
                        loss = self.loss_fn(test_pred, y)
                        test_loss += loss.item()

                        # Debug NaN/Inf loss
                        if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                            print(f"{Common.warning} Loss is NaN/Inf at batch {batch}. Skipping...")
                            continue

                    # Calculate and accumulate accuracy
                    test_pred = test_pred.float() # Convert to float for stability
                    test_pred_class = test_pred.argmax(dim=1)
                    test_acc += Common.calculate_accuracy(y, test_pred_class) #((test_pred_class == y).sum().item()/len(test_pred))

                    # Collect outputs for fpr-at-recall calculation
                    all_preds.append(torch.softmax(test_pred, dim=1).detach().cpu())
                    all_labels.append(y.detach().cpu())

            # Adjust metrics to get average loss and accuracy per batch 
            test_loss /= len_dataloader
            test_acc /= len_dataloader

            # Final FPR calculation
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            try:    
                test_fpr = Common.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
            except Exception as e:
                logging.error(f"{Common.warning} Innacurate calculation of final FPR at recall: {e}")
                test_fpr = 1.0
            try:    
                test_pauc = Common.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
            except Exception as e:
                logging.error(f"{Common.warning} Innacurate calculation of final pAUC at recall: {e}")
                test_pauc = 0.0
        
        # Otherwise set params with initial values
        else:
            test_loss, test_acc, test_fpr, test_pauc = self.best_test_loss, self.best_test_acc, self.best_test_fpr, self.best_test_pauc

        return test_loss, test_acc, test_fpr, test_pauc

    
    def display_results(
        self,
        epoch,
        max_epochs,
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

        # Retrieve the learning rate
        if self.scheduler is None or isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr = self.optimizer.param_groups[0]['lr']
        else:
            lr = self.scheduler.get_last_lr()[0]
        
        # Print results
        print(
            f"{Common.colors['BLACK']}Epoch: {epoch+1}/{max_epochs} | "
            f"{Common.colors['BLUE']}Train: {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}loss: {train_loss:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}acc: {train_acc:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}fpr: {train_fpr:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}pauc: {train_pauc:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}time: {Common.sec_to_min_sec(train_epoch_time)} {Common.colors['BLACK']}| "            
            f"{Common.colors['BLUE']}lr: {lr:.10f}"
        )
        if self.apply_validation:
            print(
                f"{Common.colors['BLACK']}Epoch: {epoch+1}/{max_epochs} | "
                f"{Common.colors['ORANGE']}Test:  {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}loss: {test_loss:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}acc: {test_acc:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}fpr: {test_fpr:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}pauc: {test_pauc:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}time: {Common.sec_to_min_sec(test_epoch_time)} {Common.colors['BLACK']}| "            
                f"{Common.colors['ORANGE']}lr: {lr:.10f}"
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
            if self.apply_validation:
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
                writer.add_scalars(
                    main_tag="Loss", 
                    tag_scalar_dict={"train_loss": train_loss},
                    global_step=epoch)
                writer.add_scalars(
                    main_tag="Accuracy", 
                    tag_scalar_dict={"train_acc": train_acc},
                    global_step=epoch)
                writer.add_scalars(
                    main_tag=f"FPR at {recall_threshold * 100}% recall", 
                    tag_scalar_dict={"train_fpr": train_fpr}, 
                    global_step=epoch)
                writer.add_scalars(
                    main_tag=f"pAUC above {recall_threshold_pauc * 100}% recall", 
                    tag_scalar_dict={"train_pauc": train_pauc}, 
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
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_loss"], label="test_loss")
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot accuracy
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, self.results["train_acc"], label="train_accuracy")
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_acc"], label="test_accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            # Plot FPR at recall
            plt.subplot(1, n_plots, 3)
            plt.plot(range_epochs, self.results["train_fpr"], label="train_fpr_at_recall")
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_fpr"], label="test_fpr_at_recall")
            plt.title(f"FPR at {recall_threshold * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot pAUC at recall
            plt.subplot(1, n_plots, 4)
            plt.plot(range_epochs, self.results["train_pauc"], label="train_pauc_at_recall")
            if self.apply_validation:
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
            if self.apply_validation and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Check whether scheduler is configured for "min" or "max"
                if self.scheduler.mode == "min" and test_loss is not None:
                    self.scheduler.step(test_loss)  # Minimize test_loss
                elif self.scheduler.mode == "max" and test_acc is not None:
                    self.scheduler.step(test_acc)  # Maximize test_accuracy
                else:
                    raise ValueError(
                        f"{Common.error}The scheduler requires either `test_loss` or `test_acc` "
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
            raise ValueError(f"{Common.error}'epoch' must be provided when mode includes 'all' or 'last'.")

        # Save model according criteria

        # Helper functions
        def remove_previous_best(model_name):
            """Removes previously saved best model files."""
            file_to_remove = glob.glob(os.path.join(self.target_dir, model_name.replace(".", "_epoch*.")))
            for f in file_to_remove:
                os.remove(f)

        def save_model(model_name):
            """Helper function to save the model."""
            self.save(model=self.model, target_dir=self.target_dir, model_name=model_name.replace(".", f"_epoch{epoch+1}."))

        if self.save_best_model:            
            for mode in self.mode:
                # Loss criterion
                if mode == "loss":
                    if test_loss is None:
                        raise ValueError(f"{Common.error}'test_loss' must be provided when mode is 'loss'.")
                    if test_loss < self.best_test_loss:
                        remove_previous_best(self.model_name_loss)
                        self.best_test_loss = test_loss
                        if self.keep_best_models_in_memory:
                            self.model_loss.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_loss)
                # Accuracy criterion    
                elif mode == "acc":
                    if test_acc is None:
                        raise ValueError(f"{Common.error}'test_acc' must be provided when mode is 'acc'.")
                    if test_acc > self.best_test_acc:
                        remove_previous_best(self.model_name_acc)
                        self.best_test_acc = test_acc
                        if self.keep_best_models_in_memory:
                            self.model_acc.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_acc)
                # FPR criterion
                elif mode == "fpr":
                    if test_fpr is None:
                        raise ValueError(f"{Common.error}'test_fpr' must be provided when mode is 'fpr'.")
                    if test_fpr < self.best_test_fpr:
                        remove_previous_best(self.model_name_fpr)
                        self.best_test_fpr = test_fpr
                        if self.keep_best_models_in_memory:
                            self.model_fpr.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_fpr)
                # pAUC criterion    
                elif mode == "pauc":
                    if test_pauc is None:
                        raise ValueError(f"{Common.error}'test_pauc' must be provided when mode is 'pauc'.")
                    if test_pauc > self.best_test_pauc:
                        remove_previous_best(self.model_name_pauc)
                        self.best_test_pauc = test_pauc
                        if self.keep_best_models_in_memory:
                            self.model_pauc.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_pauc)
                # Last-epoch criterion
                elif mode == "last":
                    remove_previous_best(self.model_name)
                    save_model(self.model_name)
                # All epochs
                elif mode == "all":
                    if self.keep_best_models_in_memory:
                        if isinstance(self.model_epoch, list) and epoch < len(self.model_epoch):
                            self.model_epoch[epoch].load_state_dict(self.model.state_dict())
                    save_model(self.model_name)

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
        print(f"{Common.info} Training finished! Elapsed time: {Common.sec_to_min_sec(train_time)}")
            
    # Trains and tests a Pytorch model
    def train(
        self,
        target_dir: str=None,
        model_name: str=None,
        save_best_model: Union[str, List[str]] = "last",
        keep_best_models_in_memory: bool=False,
        train_dataloader: torch.utils.data.DataLoader=None, 
        test_dataloader: torch.utils.data.DataLoader=None,
        apply_validation: bool=True,
        num_classes: int=2, 
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        scheduler: torch.optim.lr_scheduler=None,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=False,
        accumulation_steps: int=1,
        debug_mode: bool=False,
        writer=False, #: SummaryWriter=False,
        ) -> pd.DataFrame:

        """
        Trains and tests a PyTorch model for a given number of epochs.

        This function handles the training loop, evaluates the model performance on both training
        and test datasets, and stores the metrics. It also supports model saving, learning rate
        scheduling, and debugging capabilities. Optionally, it supports logging the training process
        with TensorBoard, and handles gradient accumulation and mixed precision training.

        During training, the model is evaluated after each epoch based on the provided dataloaders.
        The model can be saved at different stages (e.g., at the end of each epoch or when the model 
        achieves the best performance according to a specified metric such as loss, accuracy, or FPR).

        Args:
            target_dir (str, optional): Directory to save the trained model.
            model_name (str, optional): Name for the saved model file. Must include file extension
                such as ".pth", ".pt", ".pkl", ".h5", or ".torch".
            save_best_model (Union[str, List[str]], optional): Criterion(s) for saving the model.
                Options include:
                - "loss" (validation loss),
                - "acc" (validation accuracy),
                - "fpr" (false positive rate at recall),
                - "pauc" (partial area under the curve at recall),
                - "last" (save model at the last epoch),
                - "all" (save models for all epochs),
                - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
            keep_best_models_in_memory (bool, optional): If True, the best models are kept in memory for future inference. The model state from the last epoch will always be kept in memory.
            train_dataloader (torch.utils.data.DataLoader, optional): Dataloader for training the model.
            test_dataloader (torch.utils.data.DataLoader, optional): Dataloader for testing the model.
            apply_validation (bool, optional): Whether to apply validation after each epoch. Default is True.
            num_classes (int, optional): Number of output classes for the model (default is 2).
            optimizer (torch.optim.Optimizer, optional): Optimizer to use during training.
            loss_fn (torch.nn.Module, optional): Loss function used for training.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler to adjust learning rate during training.
            recall_threshold (float, optional): The recall threshold used to calculate the False Positive Rate (FPR). Default is 0.95.
            recall_threshold_pauc (float, optional): The recall threshold used to calculate the partial Area Under the Curve (pAUC). Default is 0.95.
            epochs (int, optional): Number of epochs to train the model. Default is 30.
            plot_curves (bool, optional): Whether to plot training and testing curves. Default is True.
            amp (bool, optional): Whether to use Automatic Mixed Precision (AMP) during training. Default is True.
            enable_clipping (bool, optional): Whether to enable gradient and model output clipping. Default is False.
            accumulation_steps (int, optional): Number of mini-batches to accumulate gradients before an optimizer step. Default is 1 (no accumulation).
            debug_mode (bool, optional): Whether to enable debug mode. If True, it may slow down the training process.
            writer (bool, optional): A TensorBoard SummaryWriter instance to log the model training results.

        Returns:
            pd.DataFrame: A dataframe containing the metrics for training and testing across all epochs.
            The dataframe will have the following columns:
            - epoch: List of epoch numbers.
            - train_loss: List of training loss values for each epoch.
            - train_acc: List of training accuracy values for each epoch.
            - test_loss: List of test loss values for each epoch.
            - test_acc: List of test accuracy values for each epoch.
            - train_time: List of training time for each epoch.
            - test_time: List of testing time for each epoch.
            - lr: List of learning rate values for each epoch.
            - train_fpr: List of False Positive Rate values for training set at recall threshold.
            - test_fpr: List of False Positive Rate values for test set at recall threshold.
            - train_pauc: List of partial AUC values for training set at recall threshold.
            - test_pauc: List of partial AUC values for test set at recall threshold.

        Example output (for 2 epochs):
        {
            epoch: [1, 2],
            train_loss: [2.0616, 1.0537],
            train_acc: [0.3945, 0.3945],
            test_loss: [1.2641, 1.5706],
            test_acc: [0.3400, 0.2973],
            train_time: [1.1234, 1.5678],
            test_time: [0.4567, 0.7890],
            lr: [0.001, 0.0005],
            train_fpr: [0.1234, 0.2345],
            test_fpr: [0.3456, 0.4567],
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
            keep_best_models_in_memory=keep_best_models_in_memory,
            apply_validation= apply_validation,
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
            debug_mode=debug_mode,
            writer=writer
            )

        # Loop through training and testing steps for a number of epochs
        for epoch in range(epochs):

            # Perform training step and time it
            train_epoch_start_time = time.time()
            train_loss, train_acc, train_fpr, train_pauc = self.train_step_v2(
                dataloader=train_dataloader,
                num_classes=num_classes,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                epoch_number=epoch,
                amp=amp,
                enable_clipping=enable_clipping,
                accumulation_steps=accumulation_steps,
                debug_mode=debug_mode
                )
            train_epoch_time = time.time() - train_epoch_start_time

            # Perform test step and time it
            test_epoch_start_time = time.time()
            test_loss, test_acc, test_fpr, test_pauc = self.test_step(
                dataloader=test_dataloader,
                num_classes=num_classes,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                epoch_number=epoch,
                amp=amp,
                enable_clipping=enable_clipping,
                debug_mode=debug_mode
                )
            test_epoch_time = time.time() - test_epoch_start_time if self.apply_validation else 0.0            

            clear_output(wait=True)

            # Show results
            self.display_results(
                epoch=epoch,
                max_epochs=epochs,
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
            # If apply_validation is enabled then upate models based on validation results
            df_results = self.update_model(
                test_loss=test_loss if self.apply_validation else train_loss,
                test_acc=test_acc if self.apply_validation else train_acc,
                test_fpr=test_fpr if self.apply_validation else train_fpr,
                test_pauc=test_pauc if self.apply_validation else train_pauc,
                epoch=epoch
                )

        # Finish training process
        train_time = time.time() - train_start_time
        self.finish_train(train_time, writer)

        return df_results

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_state: str="last",
        output_type: str="softmax",        
        ) -> torch.Tensor:

        """
        Predicts classes for a given dataset using a trained model.

        Args:
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
            dataloader (torch.utils.data.DataLoader): The dataset to predict on.
            output_type (str): The type of output to return. Either "softmax", "logits", or "argmax".            

        Returns:
            (list): All of the predicted class labels represented by prediction probabilities (softmax)
        """
 
        # Check model to use
        valid_modes =  {"loss", "acc", "fpr", "pauc", "last", "all"}
        assert model_state in valid_modes or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "fpr":
            if self.model_fpr is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
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
        model_state: str="last",
        sample_fraction: float=1.0,
        seed=42,        
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        
        """
        Predicts classes for a given dataset using a trained model and stores the per-sample results in dictionaries.

        Args:
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
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
        valid_modes =  {"loss", "acc", "fpr", "pauc", "last", "all"}
        assert model_state in valid_modes or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "fpr":
            if self.model_fpr is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]

        # Create a list of test images and checkout existence
        print(f"{Common.info} Finding all filepaths ending with '.jpg' in directory: {test_dir}")
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
    

# Training and prediction engine class
class DistillationEngine:

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
        student: torch.nn.Module=None,
        teacher: torch.nn.Module=None,
        device: str="cuda" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()

        # Initialize self variables
        self.device = device
        self.model = student
        self.model_tch = teacher
        self.save_best_model = False
        self.keep_best_models_in_memory = False
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
        self.get_predictions = Common.get_predictions
     
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

        # Check if model is provided
        if self.model is None:
            raise ValueError(f"{Common.error} Instantiate the engine by passing a PyTorch model to handle.")
        else:
            self.model.to(self.device)

    def save(
        self,
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
        print(f"{Common.info} Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)

    def load(
        self,
        target_dir: str,
        model_name: str):
        
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
        print(f"{Common.info} Loading model from: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        
        return self
    
    
    def create_writer(
        self,
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
            
        print(f"{Common.info} Created SummaryWriter, saving to: {log_dir}...")
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
            debug_mode,
            writer):
        
        """
        Prints the configuration of the training process.
        """

        print(f"{Common.info} Device: {self.device}")
        print(f"{Common.info} Epochs: {epochs}")
        print(f"{Common.info} Batch size: {batch_size}")
        print(f"{Common.info} Accumulation steps: {accumulation_steps}")
        print(f"{Common.info} Effective batch size: {batch_size * accumulation_steps}")
        print(f"{Common.info} Recall threshold - fpr: {recall_threshold}")
        print(f"{Common.info} Recall threshold - pauc: {recall_threshold_pauc}")
        print(f"{Common.info} Apply validation: {self.apply_validation}")
        print(f"{Common.info} Plot curves: {plot_curves}")
        print(f"{Common.info} Automatic Mixed Precision (AMP): {amp}")
        print(f"{Common.info} Enable clipping: {enable_clipping}")
        print(f"{Common.info} Debug mode: {debug_mode}")
        print(f"{Common.info} Enable writer: {writer}")
        print(f"{Common.info} Save model: {self.save_best_model}")
        print(f"{Common.info} Target directory: {self.target_dir}")        
        if self.save_best_model:
          # Extract base name and extension from the model name
            base_name, extension = os.path.splitext(self.model_name)
            
            # Print base name and extension
            print(f"{Common.info} Model name base: {base_name}")
            print(f"{Common.info} Model name extension: {extension}")
            
            # Iterate over modes and format model name, skipping 'last'
            for mode in self.mode:
                if mode == "last" or mode == "all":
                    # Skip adding 'last' and just use epoch in the filename
                    model_name_with_mode = f"_epoch<int>{extension}"
                else:
                    # For other modes, include mode and epoch in the filename
                    model_name_with_mode = f"_{mode}_epoch<int>{extension}"
                
                # Print the final model save path for each mode
                print(f"{Common.info} Save best model - {mode}: {base_name + model_name_with_mode}")
        if self.keep_best_models_in_memory:
            print(f"{Common.warning} Keeping best models in memory: {self.keep_best_models_in_memory} - it may slow down the training process.")
        else:
            print(f"{Common.info} Keeping best models in memory: {self.keep_best_models_in_memory}")

    def init_train(
        self,
        target_dir: str=None,
        model_name: str=None,
        apply_validation: bool=True,
        save_best_model: Union[str, List[str]] = "last",  # Allow both string and list
        keep_best_models_in_memory: bool=False,
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        scheduler: torch.optim.lr_scheduler=None,
        batch_size: int=64,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=False,
        accumulation_steps: int=1,
        debug_mode: bool=False,
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
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.
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

        # Validate keep_best_models_in_memory
        if not isinstance(keep_best_models_in_memory, (bool)):
            raise ValueError(f"{Common.error}'keep_best_models_in_memory' must be True or False.")
        else:
            self.keep_best_models_in_memory = keep_best_models_in_memory

        # Validate apply_validation
        if not isinstance(apply_validation, (bool)):
            raise ValueError(f"{Common.error}'apply_validation' must be True or False.")
        else:
            self.apply_validation = apply_validation
      
        # Validate recall_threshold
        if not isinstance(recall_threshold, (int, float)) or not (0.0 <= float(recall_threshold) <= 1.0):
            raise ValueError(f"{Common.error}'recall_threshold' must be a float between 0.0 and 1.0.")

        # Validate recall_threshold_pauc
        if not isinstance(recall_threshold_pauc, (int, float)) or not (0.0 <= float(recall_threshold_pauc) <= 1.0):
            raise ValueError(f"{Common.error}'recall_threshold_pauc' must be a float between 0.0 and 1.0.")

        # Validate accumulation_steps
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError(f"{Common.error}'accumulation_steps' must be an integer greater than or equal to 1.")

        # Validate epochs
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"{Common.error}'epochs' must be an integer greater than or equal to 1.")

        # Ensure save_best_model is correctly handled
        if save_best_model is None:
            self.save_best_model = False
            mode = []
        elif isinstance(save_best_model, (str, list)):
            self.save_best_model = True
            mode = [save_best_model] if isinstance(save_best_model, str) else save_best_model  # Ensure mode is a list
        else:
            raise ValueError(f"{Common.error}'save_best_model' must be None, a string, or a list of strings.")

        # Validate mode only if save_best_model is True
        valid_modes = {"loss", "acc", "fpr", "pauc", "last", "all"}
        if self.save_best_model:
            if not isinstance(mode, list):
                raise ValueError(f"{Common.error}'mode' must be a string or a list of strings.")

            for m in mode:
                if m not in valid_modes:
                    raise ValueError(f"{Common.error}Invalid mode value: '{m}'. Must be one of {valid_modes}")

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
            debug_mode=debug_mode,
            writer=writer
            )
        
        # Initialize optimizer, loss_fn, and scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
    
        # Initialize the best model and model_epoch list based on the specified mode.
        if self.save_best_model:
            if "loss" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_loss = copy.deepcopy(self.model)                            
                    self.model_loss.to(self.device)
                self.model_name_loss = self.model_name.replace(".", f"_loss.")
            if "acc" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_acc = copy.deepcopy(self.model)                            
                    self.model_acc.to(self.device)
                self.model_name_acc = self.model_name.replace(".", f"_acc.")
            if "fpr" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_fpr = copy.deepcopy(self.model)                            
                    self.model_fpr.to(self.device)
                self.model_name_fpr = self.model_name.replace(".", f"_fpr.")
            if "pauc" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_pauc = copy.deepcopy(self.model)                            
                    self.model_pauc.to(self.device)
                self.model_name_pauc = self.model_name.replace(".", f"_pauc.")
            if "all" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_epoch = []
                    for k in range(epochs):
                        self.model_epoch.append(copy.deepcopy(self.model))
                        self.model_epoch[k].to(self.device)
            self.best_test_loss = float("inf") 
            self.best_test_acc = 0.0
            self.best_test_fpr = float("inf")
            self.best_test_pauc = 0.0
    

    # This train step function includes gradient accumulation (experimental)
    def train_step_v2(
        self,
        dataloader_std: torch.utils.data.DataLoader, 
        dataloader_tch: torch.utils.data.DataLoader,
        num_classes: int=2,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epoch_number: int = 1,
        amp: bool=True,
        enable_clipping=False,
        accumulation_steps: int = 1,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
    
        """Trains a PyTorch model for a single epoch with gradient accumulation.

        Args:
            dataloader_std: A DataLoader instance for the student model to be tested on.
            dataloader_tch: A DataLoader instance for the teacheer model.
            num_classes: Number of classes.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1)
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            epoch_number: Epoch number.
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            accumulation_steps: Number of mini-batches to accumulate gradients before an optimizer step.
                If batch size is 64 and accumulation_steps is 4, gradients are accumulated for 256 mini-batches before an optimizer step.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, train_fpr, train_pauc). For example: (0.1112, 0.8743, 0.01123, 0.15561).
        """

        print(f"{Common.info} Training epoch {epoch_number+1}...")


        # Put student model in train mode
        self.model.train()
        self.model.to(self.device)

        # Put teacher model in evaluation mode
        self.model_tch.eval()
        self.model_tch.to(self.device)

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0    
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting
        for (batch, (X, y)), (_, (X_tch, _)) in tqdm(zip(enumerate(dataloader_std), enumerate(dataloader_tch)), total=len(dataloader_std)):
            
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)
            X_tch = X_tch.to(self.device)

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.get_predictions(self.model(X))
                    y_pred_tch = self.get_predictions(self.model_tch(X_tch))

                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            print(f"{Common.warning} y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            print(f"{Common.warning} y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue
                    
                    # Calculate loss, normalize by accumulation steps
                    loss = self.loss_fn(y_pred, y_pred_tch, y) / accumulation_steps
                
                    # Check for NaN or Inf in loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"{Common.warning} Loss is NaN or Inf at batch {batch}. Skipping...")
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
                y_pred = self.get_predictions(self.model(X))
                y_pred_tch = self.get_predictions(self.model_tch(X_tch))
                
                # Calculate loss, normalize by accumulation steps
                loss = self.loss_fn(y_pred, y_pred_tch, y) / accumulation_steps

                # Backward pass
                loss.backward()

            # Gradient cliping
            if enable_clipping:
                # Apply clipping if needed
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Perform optimizer step and clear gradients every accumulation_steps
            if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(dataloader_std):

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
                                    print(f"{Common.warning} NaN or Inf gradient detected in {name} at batch {batch}")
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
            y_pred = y_pred.float() # Convert to float for stability
            y_pred_class = y_pred.argmax(dim=1)
            train_acc += Common.calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item() / len(y_pred)

            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader_std)
        train_acc = train_acc / len(dataloader_std)

        # Final FPR calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        try:    
            train_fpr = Common.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
        except Exception as e:
            logging.error(f"{Common.warning} Innacurate calculation of final FPR at recall: {e}")
            train_fpr = 1.0
        try:    
            train_pauc = Common.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
        except Exception as e:
            logging.error(f"{Common.warning} Innacurate calculation of final pAUC at recall: {e}")
            train_pauc = 0.0

        return train_loss, train_acc, train_fpr, train_pauc

    def test_step(
        self,
        dataloader_std: torch.utils.data.DataLoader,
        dataloader_tch: torch.utils.data.DataLoader,
        num_classes: int=2,
        recall_threshold: float = 0.95,
        recall_threshold_pauc: float = 0.95,
        epoch_number: int = 1,
        amp: bool = True,
        debug_mode: bool = False,
        enable_clipping: bool = False
    ) -> Tuple[float, float, float]:
        
        """Tests a PyTorch model for a single epoch.

        Args:
            dataloader_std: A DataLoader instance for the student model to be tested on.
            dataloader_tch: A DataLoader instance for the teacheer model.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1).
            recall_threshold_pauc: The recall threshold for pAUC computation (between 0 and 1)
            epoch_number: Epoch number.
            amp: Whether to use Automatic Mixed Precision for inference.
            debug_mode: Enables logging for debugging purposes.
            enable_clipping: Enables NaN/Inf value clipping for test predictions.

        Returns:
            A tuple of test loss, test accuracy, FPR-at-recall, and pAUC-at-recall metrics.
        """

        # Execute the test step is apply_validation is enabled
        if self.apply_validation:

            print(f"{Common.info} Validating epoch {epoch_number+1}...")

            # Put the student model in eval mode
            self.model.eval() 
            self.model.to(self.device)

            # Put the teacher model in eval mode
            self.model_tch.eval()
            self.model_tch.to(self.device)

            # Setup test loss and test accuracy values
            test_loss, test_acc = 0, 0
            all_preds = []
            all_labels = []

            # Turn on inference context manager
            with torch.inference_mode():
                # Loop through DataLoader batches
                # for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), colour='#FF9E2C'):
                for (batch, (X, y)), (_, (X_tch, _)) in tqdm(zip(enumerate(dataloader_std), enumerate(dataloader_tch)), total=len(dataloader_std), colour='#FF9E2C'):
                    #, desc=f"Validating epoch {epoch_number}..."):
                    # Send data to target device
                    X, y = X.to(self.device), y.to(self.device)
                    X_tch = X_tch.to(self.device)

                    # Enable AMP if specified
                    with torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext():
                        test_pred = self.get_predictions(self.model(X))
                        test_pred_tch = self.get_predictions(self.model_tch(X_tch))

                        # Check for NaN/Inf in predictions
                        if torch.isnan(test_pred).any() or torch.isinf(test_pred).any():
                            if enable_clipping:
                                print(f"{Common.warning} Predictions contain NaN/Inf at batch {batch}. Applying clipping...")
                                test_pred = torch.nan_to_num(
                                    test_pred,
                                    nan=torch.mean(test_pred).item(),
                                    posinf=torch.max(test_pred).item(),
                                    neginf=torch.min(test_pred).item()
                                )
                            else:
                                print(f"{Common.warning} Predictions contain NaN/Inf at batch {batch}. Skipping batch...")
                                continue

                        # Calculate and accumulate loss
                        loss = self.loss_fn(test_pred, test_pred_tch, y)
                        test_loss += loss.item()

                        # Debug NaN/Inf loss
                        if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                            print(f"{Common.warning} Loss is NaN/Inf at batch {batch}. Skipping...")
                            continue

                    # Calculate and accumulate accuracy
                    test_pred = test_pred.float() # Convert to float for stability
                    test_pred_class = test_pred.argmax(dim=1)
                    test_acc += Common.calculate_accuracy(y, test_pred_class) #((test_pred_class == y).sum().item()/len(test_pred))

                    # Collect outputs for fpr-at-recall calculation
                    all_preds.append(torch.softmax(test_pred, dim=1).detach().cpu())
                    all_labels.append(y.detach().cpu())

            # Adjust metrics to get average loss and accuracy per batch 
            test_loss = test_loss / len(dataloader_std)
            test_acc = test_acc / len(dataloader_std)

            # Final FPR calculation
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            try:    
                test_fpr = Common.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
            except Exception as e:
                logging.error(f"{Common.warning} Innacurate calculation of final FPR at recall: {e}")
                test_fpr = 1.0
            try:    
                test_pauc = Common.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
            except Exception as e:
                logging.error(f"{Common.warning} Innacurate calculation of final pAUC at recall: {e}")
                test_pauc = 0.0
        
        # Otherwise set params with initial values
        else:
            test_loss, test_acc, test_fpr, test_pauc = self.best_test_loss, self.best_test_acc, self.best_test_fpr, self.best_test_pauc

        return test_loss, test_acc, test_fpr, test_pauc

    
    def display_results(
        self,
        epoch,
        max_epochs,
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

        # Retrieve the learning rate
        if self.scheduler is None or isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr = self.optimizer.param_groups[0]['lr']
        else:
            lr = self.scheduler.get_last_lr()[0]
        
        # Print results
        print(
            f"{Common.colors['BLACK']}Epoch: {epoch+1}/{max_epochs} | "
            f"{Common.colors['BLUE']}Train: {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}loss: {train_loss:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}acc: {train_acc:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}fpr: {train_fpr:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}pauc: {train_pauc:.4f} {Common.colors['BLACK']}| "
            f"{Common.colors['BLUE']}time: {Common.sec_to_min_sec(train_epoch_time)} {Common.colors['BLACK']}| "            
            f"{Common.colors['BLUE']}lr: {lr:.10f}"
        )
        if self.apply_validation:
            print(
                f"{Common.colors['BLACK']}Epoch: {epoch+1}/{max_epochs} | "
                f"{Common.colors['ORANGE']}Test:  {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}loss: {test_loss:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}acc: {test_acc:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}fpr: {test_fpr:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}pauc: {test_pauc:.4f} {Common.colors['BLACK']}| "
                f"{Common.colors['ORANGE']}time: {Common.sec_to_min_sec(test_epoch_time)} {Common.colors['BLACK']}| "            
                f"{Common.colors['ORANGE']}lr: {lr:.10f}"
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
            if self.apply_validation:
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
                writer.add_scalars(
                    main_tag="Loss", 
                    tag_scalar_dict={"train_loss": train_loss},
                    global_step=epoch)
                writer.add_scalars(
                    main_tag="Accuracy", 
                    tag_scalar_dict={"train_acc": train_acc},
                    global_step=epoch)
                writer.add_scalars(
                    main_tag=f"FPR at {recall_threshold * 100}% recall", 
                    tag_scalar_dict={"train_fpr": train_fpr}, 
                    global_step=epoch)
                writer.add_scalars(
                    main_tag=f"pAUC above {recall_threshold_pauc * 100}% recall", 
                    tag_scalar_dict={"train_pauc": train_pauc}, 
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
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_loss"], label="test_loss")
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot accuracy
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, self.results["train_acc"], label="train_accuracy")
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_acc"], label="test_accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            # Plot FPR at recall
            plt.subplot(1, n_plots, 3)
            plt.plot(range_epochs, self.results["train_fpr"], label="train_fpr_at_recall")
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_fpr"], label="test_fpr_at_recall")
            plt.title(f"FPR at {recall_threshold * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot pAUC at recall
            plt.subplot(1, n_plots, 4)
            plt.plot(range_epochs, self.results["train_pauc"], label="train_pauc_at_recall")
            if self.apply_validation:
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
            if self.apply_validation and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Check whether scheduler is configured for "min" or "max"
                if self.scheduler.mode == "min" and test_loss is not None:
                    self.scheduler.step(test_loss)  # Minimize test_loss
                elif self.scheduler.mode == "max" and test_acc is not None:
                    self.scheduler.step(test_acc)  # Maximize test_accuracy
                else:
                    raise ValueError(
                        f"{Common.error}The scheduler requires either `test_loss` or `test_acc` "
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
            raise ValueError(f"{Common.error}'epoch' must be provided when mode includes 'all' or 'last'.")

        # Save model according criteria

        # Helper functions
        def remove_previous_best(model_name):
            """Removes previously saved best model files."""
            file_to_remove = glob.glob(os.path.join(self.target_dir, model_name.replace(".", "_epoch*.")))
            for f in file_to_remove:
                os.remove(f)

        def save_model(model_name):
            """Helper function to save the model."""
            self.save(model=self.model, target_dir=self.target_dir, model_name=model_name.replace(".", f"_epoch{epoch+1}."))

        if self.save_best_model:            
            for mode in self.mode:
                # Loss criterion
                if mode == "loss":
                    if test_loss is None:
                        raise ValueError(f"{Common.error}'test_loss' must be provided when mode is 'loss'.")
                    if test_loss < self.best_test_loss:
                        remove_previous_best(self.model_name_loss)
                        self.best_test_loss = test_loss
                        if self.keep_best_models_in_memory:
                            self.model_loss.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_loss)
                # Accuracy criterion    
                elif mode == "acc":
                    if test_acc is None:
                        raise ValueError(f"{Common.error}'test_acc' must be provided when mode is 'acc'.")
                    if test_acc > self.best_test_acc:
                        remove_previous_best(self.model_name_acc)
                        self.best_test_acc = test_acc
                        if self.keep_best_models_in_memory:
                            self.model_acc.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_acc)
                # FPR criterion
                elif mode == "fpr":
                    if test_fpr is None:
                        raise ValueError(f"{Common.error}'test_fpr' must be provided when mode is 'fpr'.")
                    if test_fpr < self.best_test_fpr:
                        remove_previous_best(self.model_name_fpr)
                        self.best_test_fpr = test_fpr
                        if self.keep_best_models_in_memory:
                            self.model_fpr.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_fpr)
                # pAUC criterion    
                elif mode == "pauc":
                    if test_pauc is None:
                        raise ValueError(f"{Common.error}'test_pauc' must be provided when mode is 'pauc'.")
                    if test_pauc > self.best_test_pauc:
                        remove_previous_best(self.model_name_pauc)
                        self.best_test_pauc = test_pauc
                        if self.keep_best_models_in_memory:
                            self.model_pauc.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_pauc)
                # Last-epoch criterion
                elif mode == "last":
                    remove_previous_best(self.model_name)
                    save_model(self.model_name)
                # All epochs
                elif mode == "all":
                    if self.keep_best_models_in_memory:
                        if isinstance(self.model_epoch, list) and epoch < len(self.model_epoch):
                            self.model_epoch[epoch].load_state_dict(self.model.state_dict())
                    save_model(self.model_name)

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
        print(f"{Common.info} Training finished! Elapsed time: {Common.sec_to_min_sec(train_time)}")

    
    # Trains and tests a Pytorch model
    def train(
        self,
        target_dir: str=None,
        model_name: str=None,
        save_best_model: Union[str, List[str]] = "last",
        keep_best_models_in_memory: bool=False,
        train_dataloader_std: torch.utils.data.DataLoader=None,
        train_dataloader_tch: torch.utils.data.DataLoader=None, 
        test_dataloader_std: torch.utils.data.DataLoader=None,
        test_dataloader_tch: torch.utils.data.DataLoader=None,
        apply_validation: bool=True,
        num_classes: int=2, 
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        scheduler: torch.optim.lr_scheduler=None,
        recall_threshold: float=0.95,
        recall_threshold_pauc: float=0.95,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=False,
        accumulation_steps: int=1,
        debug_mode: bool=False,
        writer=False, #: SummaryWriter=False,
        ) -> pd.DataFrame:

            
        """Trains and tests a PyTorch model using both student and teacher models.

        The function passes the student and teacher models through the train_step() 
        and test_step() functions for a number of epochs, training and testing the 
        models in the same epoch loop.

        Calculates, prints, and stores evaluation metrics throughout. Optionally, 
        stores the metrics in a writer log directory if provided.

        Args:
            target_dir: A directory for saving the model to.
            model_name: A filename for the saved model. Should include ".pth", 
                        ".pt", ".pkl", ".h5", or ".torch" as the file extension.
            save_best_model (Union[str, List[str]]): Criterion mode for saving the model: 
            - "loss" (validation loss)
            - "acc" (validation accuracy)
            - "fpr" (false positive rate at recall)
            - "pauc" (partial area under the curve at recall)
            - "last" (last epoch)
            - "all" (save models for all epochs)
            - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
            keep_best_models_in_memory: If True, keeps the best models in memory for future use during inference. The model state at the last epoch will always be kept in memory.
            train_dataloader_std: A DataLoader instance for the student model to be trained on.
            train_dataloader_tch: A DataLoader instance for the teacher model to be trained on.
            test_dataloader_std: A DataLoader instance for the student model to be tested on.
            test_dataloader_tch: A DataLoader instance for the teacher model to be tested on.
            apply_validation:
            - If set to True, the model's performance is evaluated on the validation dataset after each epoch,
            helping to detect overfitting and guide potential adjustments in hyperparameters.
            - If set to False, validation is skipped, which reduces computational cost and speeds up training,
            but at the risk of overfitting.
            - Default: True
            num_classes: The number of classes for the classification task.
            optimizer: A PyTorch optimizer to minimize the loss function.
            loss_fn: A PyTorch loss function to calculate loss for both the student and teacher datasets.
            scheduler: A PyTorch learning rate scheduler to adjust the learning rate during training.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1). 
            recall_threshold_pauc: The recall threshold at which to calculate the pAUC score (between 0 and 1).
            epochs: The number of epochs to train the model.
            plot_curves: Whether to plot the training and testing curves.
            amp: Whether to use Automatic Mixed Precision (AMP) for training.
            enable_clipping: Whether to apply clipping to gradients and model outputs.
            accumulation_steps: Number of mini-batches to accumulate gradients before performing an optimizer step.
            debug_mode: Whether to enable the debug mode, which may slow down training.
            writer: A SummaryWriter instance to log model metrics.

        Returns:
            A dataframe of training and testing loss, accuracy, FPR at recall, and pAUC 
            for each epoch. The dataframe contains the following columns:
                - epoch: The epoch number.
                - train_loss: The training loss for each epoch.
                - train_acc: The training accuracy for each epoch.
                - test_loss: The testing loss for each epoch.
                - test_acc: The testing accuracy for each epoch.
                - train_time: The time taken for training each epoch.
                - test_time: The time taken for testing each epoch.
                - lr: The learning rate for each epoch.
                - train_fpr: The training FPR at recall for each epoch.
                - test_fpr: The testing FPR at recall for each epoch.
                - train_pauc: The training pAUC for each epoch.
                - test_pauc: The testing pAUC for each epoch.

            Example Output (for 2 epochs):
                {
                    epoch: [1, 2],
                    train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973],
                    train_time: [1.1234, 1.5678],
                    test_time: [0.4567, 0.7890],
                    lr: [0.001, 0.0005],
                    train_fpr: [0.1234, 0.2345],
                    test_fpr: [0.3456, 0.4567],
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
            keep_best_models_in_memory=keep_best_models_in_memory,
            apply_validation= apply_validation,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            batch_size=train_dataloader_std.batch_size,
            recall_threshold=recall_threshold,
            recall_threshold_pauc=recall_threshold_pauc,
            epochs=epochs, 
            plot_curves=plot_curves,
            amp=amp,
            enable_clipping=enable_clipping,
            accumulation_steps=accumulation_steps,
            debug_mode=debug_mode,
            writer=writer
            )

        # Loop through training and testing steps for a number of epochs
        for epoch in range(epochs):

            # Perform training step and time it
            train_epoch_start_time = time.time()
            train_loss, train_acc, train_fpr, train_pauc = self.train_step_v2(
                dataloader_std=train_dataloader_std,
                dataloader_tch=train_dataloader_tch,
                num_classes=num_classes,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                epoch_number=epoch,
                amp=amp,
                enable_clipping=enable_clipping,
                accumulation_steps=accumulation_steps,
                debug_mode=debug_mode
                )
            train_epoch_time = time.time() - train_epoch_start_time

            # Perform test step and time it
            test_epoch_start_time = time.time()
            test_loss, test_acc, test_fpr, test_pauc = self.test_step(
                dataloader_std=test_dataloader_std,
                dataloader_tch=test_dataloader_tch,
                num_classes=num_classes,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                epoch_number=epoch,
                amp=amp,
                enable_clipping=enable_clipping,
                debug_mode=debug_mode
                )
            test_epoch_time = time.time() - test_epoch_start_time if self.apply_validation else 0.0            

            clear_output(wait=True)

            # Show results
            self.display_results(
                epoch=epoch,
                max_epochs=epochs,
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
            # If apply_validation is enabled then upate models based on validation results
            df_results = self.update_model(
                test_loss=test_loss if self.apply_validation else train_loss,
                test_acc=test_acc if self.apply_validation else train_acc,
                test_fpr=test_fpr if self.apply_validation else train_fpr,
                test_pauc=test_pauc if self.apply_validation else train_pauc,
                epoch=epoch
                )

        # Finish training process
        train_time = time.time() - train_start_time
        self.finish_train(train_time, writer)

        return df_results

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_state: str="last",
        output_type: str="softmax",        
        ) -> torch.Tensor:

        """
        Predicts classes for a given dataset using a trained model.

        Args:
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
            dataloader (torch.utils.data.DataLoader): The dataset to predict on.
            output_type (str): The type of output to return. Either "softmax", "logits", or "argmax".            

        Returns:
            (list): All of the predicted class labels represented by prediction probabilities (softmax)
        """
 
        # Check model to use
        valid_modes =  {"loss", "acc", "fpr", "pauc", "last", "all"}
        assert model_state in valid_modes or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "fpr":
            if self.model_fpr is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
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
        model_state: str="last",
        sample_fraction: float=1.0,
        seed=42,        
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        
        """
        Predicts classes for a given dataset using a trained model and stores the per-sample results in dictionaries.

        Args:
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
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
        valid_modes =  {"loss", "acc", "fpr", "pauc", "last", "all"}
        assert model_state in valid_modes or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "fpr":
            if self.model_fpr is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                print(f"{Common.info} Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"{Common.info} Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]

        # Create a list of test images and checkout existence
        print(f"{Common.info} Finding all filepaths ending with '.jpg' in directory: {test_dir}")
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