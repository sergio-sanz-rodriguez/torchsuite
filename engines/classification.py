"""
Contains classes for training and testing a PyTorch model for classification tasks.  
"""

import os
import glob
import torch
import torchvision
import torchaudio
import random
import time
import numpy as np
import pandas as pd
import copy
import warnings
import re
from datetime import datetime
from typing import Tuple, Dict, Any, List, Union, Optional, Callable
from tqdm.auto import tqdm 
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
from .common import Common, Colors

import warnings
warnings.filterwarnings("ignore")

            
# Training and prediction engine class
class ClassificationEngine(Common):

    """
    A class to handle training, evaluation, and predictions for a PyTorch model.

    This class provides functionality to manage the training and evaluation lifecycle
    of a PyTorch model, including saving the best model based on specific criteria.

    Args:
        model (torch.nn.Module, optional): The PyTorch model to handle. Must be instantiated.
        color_map (dict, optional): Specifies the colors for the training and evaluation curves:
        'black', 'blue', 'orange', 'green', 'red', 'yellow', 'magenta', 'cyan', 'white',
        'light_gray', 'dark_gray', 'light_blue', 'light_green', 'light_red', 'light_yellow',
        'light_magenta', 'light_cyan'.
        log_verbose (bool, optional): if True, activate logger messages.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(
        self,
        model: torch.nn.Module=None,
        color_map: dict=None,
        log_verbose: bool=True,
        device: str="cuda" if torch.cuda.is_available() else "cpu"
        ):

        super().__init__()

        # Initialize self variables
        self.device = device
        self.model = model
        self.model_acc = None
        self.model_f1 = None
        self.model_loss = None
        self.model_fpr = None
        self.model_pauc = None
        self.model_epoch = None
        self.save_best_model = False
        self.keep_best_models_in_memory = False
        self.mode = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.model_name = None
        self.model_name_loss = None
        self.model_name_acc = None
        self.model_name_f1 = None
        self.model_name_fpr = None
        self.model_name_pauc = None
        self.squeeze_dim = False
        self.log_verbose = log_verbose

        # Initialize colors
        default_color_map = {'train': 'blue', 'test': 'orange', 'other': 'black'}

        # If the user provides a color_map, update the default with it
        if color_map is None:
            color_map = default_color_map # Use defaults if no user input
        else:
            color_map = {**default_color_map, **color_map} # Merge user input with defaults
        self.color_train = Colors.get_console_color(color_map['train'])
        self.color_test =  Colors.get_console_color(color_map['test'])
        self.color_other = Colors.get_console_color(color_map['other'])
        self.color_train_plt = Colors.get_matplotlib_color(color_map['train'])
        self.color_test_plt =  Colors.get_matplotlib_color(color_map['test'])

        # Initialize result logs
        self.results = {}

        # Check if model is provided
        if self.model is None:
            self.error(f"Instantiate the engine by passing a PyTorch model to handle.")
        else:
            self.model.to(self.device)

    def save(
        self,
        model: torch.nn.Module,
        target_dir: str,
        model_name: str
        ):

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
        if not any(model_name.endswith(ext) for ext in valid_extensions):
            self.error(f"'model_name' should end with one of {valid_extensions}.")
        #assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
        model_save_path = Path(target_dir) / model_name

        # Save the model state_dict()
        self.info(f"Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)

    def load(
        self,
        target_dir: str,
        model_name: str
        ):
        
        """Loads a PyTorch model from a target directory and optionally returns it.

        Args:
            target_dir: A directory where the model is located.
            model_name: The name of the model to load. Should include
            ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.
            return_model: Whether to return the loaded model (default: False).

        Returns:
            The loaded PyTorch model (if return_model=True).
        """

        # Define the list of valid extensions
        valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

        # Create the model path
        if not any(model_name.endswith(ext) for ext in valid_extensions):
            self.error(f"'model_name' should end with one of {valid_extensions}.")
        #assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_path = Path(target_dir) / model_name

        # Load the model
        self.info(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        
        return self
    
    
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
            debug_mode
            ):
        
        """
        Prints the configuration of the training process.
        """

        self.info(f"Device: {self.device}")
        self.info(f"Epochs: {epochs}")
        self.info(f"Batch size: {batch_size}")
        self.info(f"Accumulation steps: {accumulation_steps}")
        self.info(f"Effective batch size: {batch_size * accumulation_steps}")
        self.info(f"Recall threshold - fpr: {recall_threshold}")
        self.info(f"Recall threshold - pauc: {recall_threshold_pauc}")
        self.info(f"Apply validation: {self.apply_validation}")
        self.info(f"Plot curves: {plot_curves}")
        self.info(f"Automatic Mixed Precision (AMP): {amp}")
        self.info(f"Enable clipping: {enable_clipping}")
        self.info(f"Debug mode: {debug_mode}")        
        self.info(f"Save model: {self.save_best_model}")
        self.info(f"Target directory: {self.target_dir}")        
        if self.save_best_model:
          # Extract base name and extension from the model name
            base_name, extension = os.path.splitext(self.model_name)
            
            # Print base name and extension
            self.info(f"Model name base: {base_name}")
            self.info(f"Model name extension: {extension}")
            
            # Iterate over modes and format model name, skipping 'last'
            for mode in self.mode:
                if mode == "last" or mode == "all":
                    # Skip adding 'last' and just use epoch in the filename
                    model_name_with_mode = f"_epoch<int>{extension}"
                else:
                    # For other modes, include mode and epoch in the filename
                    model_name_with_mode = f"_{mode}_epoch<int>{extension}"
                
                # Print the final model save path for each mode
                self.info(f"Save best model - {mode}: {base_name + model_name_with_mode}")
        if self.keep_best_models_in_memory:
            self.warning(f"Keeping best models in memory: {self.keep_best_models_in_memory} - it may slow down the training process.")
        else:
            self.info(f"Keeping best models in memory: {self.keep_best_models_in_memory}")
    
    # Function to initialize the result logs
    def init_results(self):

        """Creates a empty results dictionary"""

        self.results = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "train_fpr": [],
            "train_pauc": [],
            "train_time [s]": [],
            "test_loss": [],
            "test_acc": [],
            "test_f1": [],
            "test_fpr": [],
            "test_pauc": [],
            "test_time [s]": [],
            "lr": [],
            }
    
    # Function to display and plot the results
    def display_results(
        self,
        epoch,
        max_epochs,
        train_loss,
        train_acc,
        train_f1,
        recall_threshold,
        recall_threshold_pauc,
        train_fpr,
        train_pauc,
        train_epoch_time,
        test_loss,
        test_acc,
        test_f1,
        test_fpr,
        test_pauc,
        test_epoch_time,
        plot_curves,
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
            f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
            f"{self.color_train}Train: {self.color_other}| "
            f"{self.color_train}loss: {train_loss:.4f} {self.color_other}| "
            f"{self.color_train}acc: {train_acc:.4f} {self.color_other}| "
            f"{self.color_train}f1: {train_f1:.4f} {self.color_other}| "
            f"{self.color_train}fpr: {train_fpr:.4f} {self.color_other}| "
            f"{self.color_train}pauc: {train_pauc:.4f} {self.color_other}| "
            f"{self.color_train}time: {self.sec_to_min_sec(train_epoch_time)} {self.color_other}| "            
            f"{self.color_train}lr: {lr:.10f}"
        )
        if self.apply_validation:
            print(
                f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
                f"{self.color_test}Test:  {self.color_other}| "
                f"{self.color_test}loss: {test_loss:.4f} {self.color_other}| "
                f"{self.color_test}acc: {test_acc:.4f} {self.color_other}| "
                f"{self.color_test}f1: {test_f1:.4f} {self.color_other}| "
                f"{self.color_test}fpr: {test_fpr:.4f} {self.color_other}| "
                f"{self.color_test}pauc: {test_pauc:.4f} {self.color_other}| "
                f"{self.color_test}time: {self.sec_to_min_sec(test_epoch_time)} {self.color_other}| "            
                f"{self.color_test}lr: {lr:.10f}"
            )
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)
        self.results["train_loss"].append(train_loss)
        self.results["train_acc"].append(train_acc)
        self.results["train_f1"].append(train_f1)
        self.results["test_loss"].append(test_loss)
        self.results["test_acc"].append(test_acc)
        self.results["test_f1"].append(test_f1)
        self.results["train_time [s]"].append(train_epoch_time)
        self.results["test_time [s]"].append(test_epoch_time)
        self.results["lr"].append(lr)
        self.results["train_fpr"].append(train_fpr)
        self.results["test_fpr"].append(test_fpr)
        self.results["train_pauc"].append(train_pauc)
        self.results["test_pauc"].append(test_pauc)

        # Plots training and test loss, accuracy, and fpr-at-recall curves.
        if plot_curves:
        
            n_plots = 5
            plt.figure(figsize=(25, 6))
            range_epochs = range(1, len(self.results["train_loss"])+1)

            # Plot loss
            plt.subplot(1, n_plots, 1)
            plt.plot(range_epochs, self.results["train_loss"], label="train_loss", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_loss"], label="test_loss", color=self.color_test_plt)
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot accuracy
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, self.results["train_acc"], label="train_accuracy", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_acc"], label="test_accuracy", color=self.color_test_plt)
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot f1-score
            plt.subplot(1, n_plots, 3)
            plt.plot(range_epochs, self.results["train_f1"], label="train_f1", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_f1"], label="test_f1", color=self.color_test_plt)
            plt.title("F1-Score")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            # Plot FPR at recall
            plt.subplot(1, n_plots, 4)
            plt.plot(range_epochs, self.results["train_fpr"], label="train_fpr_at_recall", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_fpr"], label="test_fpr_at_recall", color=self.color_test_plt)
            plt.title(f"FPR at {recall_threshold * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot pAUC at recall
            plt.subplot(1, n_plots, 5)
            plt.plot(range_epochs, self.results["train_pauc"], label="train_pauc_at_recall", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_pauc"], label="test_pauc_at_recall", color=self.color_test_plt)
            plt.title(f"pAUC above {recall_threshold_pauc * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            plt.show()

    def init_train(
        self,
        target_dir: str=None,
        model_name: str=None,
        dataloader: torch.utils.data.DataLoader=None,
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
        debug_mode: bool=False
        ):

        """
        Initializes the training process by setting up the required configurations, parameters, and resources.

        Args:
            target_dir (str, optional): Directory to save the models. Defaults to "models" if not provided.
            model_name (str, optional): Name of the model file to save. Defaults to the class name of the model with ".pth" extension.
            dataloader: A DataLoader instance for the model to be trained on.
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
                - "f1": saves the epoch with the highest valiation f1-score
                - "fpr": saves the eopch with the lowsest false positive rate at recall
                - "pauc": saves the epoch with the highest partial area under the curve at recall
                - "last": saves last epoch
                - "all": saves models for all epochs
                - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
                - None: the model will not be saved.

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
            self.error(f"'keep_best_models_in_memory' must be True or False.")
        else:
            self.keep_best_models_in_memory = keep_best_models_in_memory

        # Validate apply_validation
        if not isinstance(apply_validation, (bool)):
            self.error(f"'apply_validation' must be True or False.")
        else:
            self.apply_validation = apply_validation
      
        # Validate recall_threshold
        if not isinstance(recall_threshold, (int, float)) or not (0.0 <= float(recall_threshold) <= 1.0):
            self.error(f"'recall_threshold' must be a float between 0.0 and 1.0.")

        # Validate recall_threshold_pauc
        if not isinstance(recall_threshold_pauc, (int, float)) or not (0.0 <= float(recall_threshold_pauc) <= 1.0):
            self.error(f"'recall_threshold_pauc' must be a float between 0.0 and 1.0.")

        # Validate accumulation_steps
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            self.error(f"'accumulation_steps' must be an integer greater than or equal to 1.")

        # Validate epochs
        if not isinstance(epochs, int) or epochs < 1:
            self.error(f"'epochs' must be an integer greater than or equal to 1.")

        # Ensure save_best_model is correctly handled
        if save_best_model is None:
            self.save_best_model = False
            mode = []
        elif isinstance(save_best_model, (str, list)):
            self.save_best_model = True
            mode = [save_best_model] if isinstance(save_best_model, str) else save_best_model  # Ensure mode is a list
        else:
            self.error(f"'save_best_model' must be None, a string, or a list of strings.")

        # Validate mode only if save_best_model is True
        valid_modes = {"loss", "acc", "f1", "fpr", "pauc", "last", "all"}
        if self.save_best_model:
            if not isinstance(mode, list):
                self.error(f"'mode' must be a string or a list of strings.")

            for m in mode:
                if m not in valid_modes:
                    self.error(f"Invalid mode value: '{m}'. Must be one of {valid_modes}")

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
            debug_mode=debug_mode
            )
        
        # Initialize optimizer, loss_fn, scheduler, and result_log
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.init_results()

        # Set the model in train mode
        self.model.train()

        # Attempt a forward pass to check if the shape of X is compatible
        for batch, (X, y) in enumerate(dataloader):
            
            try:
                # This is where the model will "complain" if the shape is incorrect
                check = self.get_predictions(self.model(X.to(self.device)))                                

            except Exception as e:
                # If the shape is wrong, reshape X and try again
                match = re.search(r"got input of size: (\[[^\]]+\])", str(e))
                if match:
                    self.warning(f"Wrong input shape: {match.group(1)}. Attempting to reshape X.")
                else:
                    self.warning(f"Attempting to reshape X.")

                # Check the current shape of X and attempt a fix
                if X.ndimension() == 3:  # [batch_size, 1, time_steps]
                    self.squeeze_dim = True
                elif X.ndimension() == 2:  # [batch_size, time_steps]
                    pass  # No change needed
                else:
                    raise ValueError(f"Unexpected input shape after exception handling: {X.shape}")
            break
    
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
            if "f1" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_f1 = copy.deepcopy(self.model)                            
                    self.model_f1.to(self.device)
                self.model_name_f1 = self.model_name.replace(".", f"_f1.")
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
            self.best_test_f1 = 0.0
            self.best_test_fpr = float("inf")
            self.best_test_pauc = 0.0
    
    def progress_bar(
        self,
        dataloader: torch.utils.data.DataLoader,
        total: int,
        stage: str,
        epoch_number: int = 1,
        desc_length: int = 22):

        """
        Creates the tqdm progress bar for the training and validation stages.

        Args:
            dataloader: The dataloader for the current stage.
            total: The total number of batches in the dataloader.
            epoch_number: The current epoch number.
            stage: The current stage ("train" or "validate").
            desc_length: The length of the description string for the progress bar.

        Returns:
            A tqdm progress bar instance for the current stage.
        """

        train_str = f"Training epoch {epoch_number+1}"
        val_str = f"Validating epoch {epoch_number+1}"
        
        if stage == 'train':
            color = self.color_train_plt
            desc = f"Training epoch {epoch_number+1}".ljust(desc_length) + " "
        elif stage == 'validation' or stage == "test":
            color = self.color_test_plt
            desc = f"Validating epoch {epoch_number+1}".ljust(desc_length) + " "
        else:
            color = self.color_test_plt
            desc = f"Making predictions"
        progress = tqdm(enumerate(dataloader), total=total, colour=color)
        progress.set_description(desc)

        return progress

    def train_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_classes: int=2,
        recall_threshold: float=0.99,
        recall_threshold_pauc: float=0.0,
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

        # Put model in train mode
        self.model.train()
        #self.model.to(self.device) # Already done in __init__

        # Initialize the GradScaler for  Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        len_dataloader = len(dataloader)
        train_loss, train_acc, train_f1 = 0, 0, 0
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        for batch, (X, y) in self.progress_bar(
            dataloader=dataloader,
            total=len_dataloader,
            epoch_number=epoch_number, stage='train'
            ):

            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)
            X = X.squeeze(1) if self.squeeze_dim else X

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.model(X)
                    y_pred = y_pred.contiguous()

                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            self.warning(f"y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            self.warning(f"y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue

                    # Calculate  and accumulate loss
                    loss = self.loss_fn(y_pred, y)
                
                    # Check for NaN or Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.warning(f"Loss is NaN or Inf at batch {batch}. Skipping batch...")
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
                                self.warning(f"NaN or Inf gradient detected in {name} at batch {batch}.")
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
            train_acc += self.calculate_accuracy(y, y_pred_class)
            
            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len_dataloader
        train_acc = train_acc / len_dataloader

        # Final FPR calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        try:
            train_f1 = self.calculate_f1_score(all_labels, all_preds.argmax(dim=1), num_classes)
        except Exception as e:
            self.warning(f"Innacurate calculation of F1-score: {e}")
        try:    
            train_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
        except Exception as e:
            self.warning(f"Innacurate calculation of final FPR at recall: {e}")
            train_fpr = 1.0
        try:    
            train_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
        except Exception as e:
            self.warning(f"Innacurate calculation of final pAUC at recall: {e}")
            train_pauc = 0.0

        return train_loss, train_acc, train_f1, train_fpr, train_pauc

    # This train step function includes gradient accumulation (experimental)
    def train_step_v2(
        self,
        dataloader: torch.utils.data.DataLoader, 
        num_classes: int=2,
        recall_threshold: float=0.99,
        recall_threshold_pauc: float=0.0,
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

        # Put model in train mode
        self.model.train()
        #self.model.to(self.device) # Already done in __init__

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        len_dataloader = len(dataloader)
        train_loss, train_acc, train_f1 = 0, 0, 0
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting
        for batch, (X, y) in self.progress_bar(
            dataloader=dataloader,
            total=len_dataloader,
            epoch_number=epoch_number,
            stage='train'
            ):
            
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)            
            X = X.squeeze(1) if self.squeeze_dim else X

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.get_predictions(self.model(X))
                    
                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            self.warning(f"y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            self.warning(f"y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue
                    
                    # Calculate loss, normalize by accumulation steps
                    loss = self.loss_fn(y_pred, y) / accumulation_steps
                
                    # Check for NaN or Inf in loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        self.warning(f"Loss is NaN or Inf at batch {batch}. Skipping...")
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
                                    self.warning(f"NaN or Inf gradient detected in {name} at batch {batch}")
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
            train_acc += self.calculate_accuracy(y, y_pred_class)

            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len_dataloader
        train_acc = train_acc / len_dataloader

        # Final FPR calculation
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        try:
            train_f1 = self.calculate_f1_score(all_labels, all_preds.argmax(dim=1), num_classes)
        except Exception as e:
            self.warning(f"Innacurate calculation of F1-score: {e}")
        try:    
            train_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
        except Exception as e:
            self.warning(f"Innacurate calculation of final FPR at recall: {e}")
            train_fpr = 1.0
        try:    
            train_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
        except Exception as e:
            self.warning(f"Innacurate calculation of final pAUC at recall: {e}")
            train_pauc = 0.0

        return train_loss, train_acc, train_f1, train_fpr, train_pauc

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

            # Put model in eval mode
            self.model.eval() 
            #self.model.to(self.device) # Already done in __init__

            # Setup test loss and test accuracy values
            len_dataloader = len(dataloader)
            test_loss, test_acc, test_f1 = 0, 0, 0
            all_preds = []
            all_labels = []

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():        
                    for batch, (X, y) in enumerate(dataloader):
                        test_pred = self.get_predictions(self.model(X.to(self.device)))
                        break
            except RuntimeError:
                inference_context = torch.no_grad()
                self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

            # Turn on inference context manager 
            with inference_context:

                # Loop through DataLoader batches
                for batch, (X, y) in self.progress_bar(
                    dataloader=dataloader,
                    total=len_dataloader,
                    epoch_number=epoch_number,
                    stage='test'):

                    # Send data to target device
                    X, y = X.to(self.device), y.to(self.device)                    
                    X = X.squeeze(1) if self.squeeze_dim else X
                    
                    if torch.isnan(X).any() or torch.isinf(X).any():
                        self.warning(f"NaN or Inf detected in test input!")

                    # Enable AMP if specified
                    with torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext():

                         # Forward pass
                        y_pred = self.get_predictions(self.model(X))

                        # Check for NaN/Inf in predictions
                        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                            if enable_clipping:
                                self.warning(f"Predictions contain NaN/Inf at batch {batch}. Applying clipping...")
                                y_pred = torch.nan_to_num(
                                    y_pred,
                                    nan=torch.mean(y_pred).item(),
                                    posinf=torch.max(y_pred).item(),
                                    neginf=torch.min(y_pred).item()
                                )
                            else:
                                self.warning(f"Predictions contain NaN/Inf at batch {batch}. Skipping batch...")
                                continue

                        # Calculate and accumulate loss
                        loss = self.loss_fn(y_pred, y)
                        test_loss += loss.item()

                        # Debug NaN/Inf loss
                        if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                            self.warning(f"Loss is NaN/Inf at batch {batch}. Skipping...")
                            continue

                    # Calculate and accumulate accuracy
                    y_pred = y_pred.float() # Convert to float for stability
                    y_pred_class = y_pred.argmax(dim=1)
                    test_acc += self.calculate_accuracy(y, y_pred_class)

                    # Collect outputs for fpr-at-recall calculation
                    all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
                    all_labels.append(y.detach().cpu())

            # Adjust metrics to get average loss and accuracy per batch 
            test_loss = test_loss / len_dataloader
            test_acc = test_acc / len_dataloader

            # Final FPR calculation
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            try:
                test_f1 = self.calculate_f1_score(all_labels, all_preds.argmax(dim=1), num_classes)
            except Exception as e:
                self.warning(f"Innacurate calculation of F1-score: {e}")
            try:    
                test_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
            except Exception as e:
                self.warning(f"Innacurate calculation of final FPR at recall: {e}")
                test_fpr = 1.0
            try:    
                test_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
            except Exception as e:
                self.warning(f"Innacurate calculation of final pAUC at recall: {e}")
                test_pauc = 0.0
        
        # Otherwise set params with initial values
        else:
            test_loss, test_acc, test_f1, test_fpr, test_pauc = self.best_test_loss, self.best_test_acc, self.best_test_f1, self.best_test_fpr, self.best_test_pauc

        return test_loss, test_acc, test_f1, test_fpr, test_pauc

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
                    self.error(
                        f"The scheduler requires either `test_loss` or `test_acc` "
                        "depending on its mode ('min' or 'max')."
                        )
            else:
                self.scheduler.step()  # For other schedulers
    
    # Updates and saves the best model and model_epoch list based on the specified mode.
    def update_model(
        self,
        test_loss: float = None,
        test_acc: float = None,
        test_f1: float = None,
        test_fpr: float = None,
        test_pauc: float = None,
        epoch: int = None,
    ) -> pd.DataFrame:

        """
        Updates and saves the best model and model_epoch list based on the specified mode(s), as well as the last-epoch model.

        Parameters:
        - test_loss (float, optional): Test loss for the current epoch (used in "loss" mode).
        - test_acc (float, optional): Test accuracy for the current epoch (used in "acc" mode).
        - test_f1 (float, optional): Test F1_score for the current epoch (used in "f1" mode).
        - test_fpr (float, optional): Test false positive rate at the specified recall (used in "fpr" mode).
        - test_pauc (float, optional): Test pAUC at the specified recall (used in "pauc" mode).
        - epoch (int, optional): Current epoch index, used for naming models when saving all epochs in "all" mode.

        Functionality:
        - Saves the last-epoch model.
        - Saves the logs (self.results).
        - Saves the best-performing model during training based on the specified evaluation mode.
        - If mode is "all", saves the model for every epoch.
        - Updates `self.model_<loss, acc, fpr, pauc, epoch>` accordingly.

        Returns:
            A dataframe of training and testing metrics for each epoch.
        """

        if isinstance(self.mode, str):
            self.mode = [self.mode]  # Ensure self.mode is always a list

        if epoch is None:
            self.error(f"'epoch' must be provided when mode includes 'all' or 'last'.")

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
                        self.error(f"'test_loss' must be provided when mode is 'loss'.")
                    if test_loss < self.best_test_loss:
                        remove_previous_best(self.model_name_loss)
                        self.best_test_loss = test_loss
                        if self.keep_best_models_in_memory:
                            self.model_loss.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_loss)
                # Accuracy criterion    
                elif mode == "acc":
                    if test_acc is None:
                        self.error(f"'test_acc' must be provided when mode is 'acc'.")
                    if test_acc > self.best_test_acc:
                        remove_previous_best(self.model_name_acc)
                        self.best_test_acc = test_acc
                        if self.keep_best_models_in_memory:
                            self.model_acc.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_acc)
                # F1-score criterion    
                elif mode == "f1":
                    if test_f1 is None:
                        self.error(f"'test_f1' must be provided when mode is 'f1'.")
                    if test_f1 > self.best_test_f1:
                        remove_previous_best(self.model_name_f1)
                        self.best_test_f1 = test_f1
                        if self.keep_best_models_in_memory:
                            self.model_f1.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_f1)
                # FPR criterion
                elif mode == "fpr":
                    if test_fpr is None:
                        self.error(f"'test_fpr' must be provided when mode is 'fpr'.")
                    if test_fpr < self.best_test_fpr:
                        remove_previous_best(self.model_name_fpr)
                        self.best_test_fpr = test_fpr
                        if self.keep_best_models_in_memory:
                            self.model_fpr.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_fpr)
                # pAUC criterion    
                elif mode == "pauc":
                    if test_pauc is None:
                        self.error(f"'test_pauc' must be provided when mode is 'pauc'.")
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
        ):

        """
        Finalizes the training process by showing the elapsed time.
        
        Args:
            train_time: Elapsed time.
        """

        # Print elapsed time
        self.info(f"Training finished! Elapsed time: {self.sec_to_min_sec(train_time)}")
            
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
        recall_threshold: float=0.99,
        recall_threshold_pauc: float=0.0,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=False,
        accumulation_steps: int=1,
        debug_mode: bool=False,
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
                - "f1" (validation F1-score),
                - "fpr" (false positive rate at recall),
                - "pauc" (partial area under the curve at recall),
                - "last" (save model at the last epoch),
                - "all" (save models for all epochs),
                - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
            keep_best_models_in_memory (bool, optional): If True, the best models are kept in memory for future inference. The model state from the last epoch will always be kept in memory.
            train_dataloader (torch.utils.data.DataLoader, optional): Dataloader for training the model.
            test_dataloader (torch.utils.data.DataLoader, optional): Dataloader for testing/validating the model.
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

        Returns:
            pd.DataFrame: A dataframe containing the metrics for training and testing across all epochs.
            The dataframe will have the following columns:
            - epoch: List of epoch numbers.
            - train_loss: List of training loss values for each epoch.
            - train_acc: List of training accuracy values for each epoch.
            - train_f1: List of training F1-score values for each epoch.
            - test_loss: List of test loss values for each epoch.
            - test_acc: List of test accuracy values for each epoch.
            - test_f1: List of test F1-score values for each epoch.
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
            train_f1: [0.4415, 0.5015],
            test_loss: [1.2641, 1.5706],
            test_acc: [0.3400, 0.2973],
            test_f1: [0.4174, 0.3481],
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
            dataloader=train_dataloader,
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
            )

        # Loop through training and testing steps for a number of epochs
        for epoch in range(epochs):

            # Perform training step and time it
            train_epoch_start_time = time.time()
            train_loss, train_acc, train_f1, train_fpr, train_pauc = self.train_step_v2(
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
            test_loss, test_acc, test_f1, test_fpr, test_pauc = self.test_step(
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
                train_f1=train_f1,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                train_fpr=train_fpr,
                train_pauc=train_pauc,
                train_epoch_time=train_epoch_time,
                test_loss=test_loss,
                test_acc=test_acc,
                test_f1=test_f1,
                test_fpr=test_fpr,
                test_pauc=test_pauc,
                test_epoch_time=test_epoch_time,
                plot_curves=plot_curves,
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
                test_f1=test_f1 if self.apply_validation else train_f1,
                test_fpr=test_fpr if self.apply_validation else train_fpr,
                test_pauc=test_pauc if self.apply_validation else train_pauc,
                epoch=epoch
                )

        # Finish training process
        train_time = time.time() - train_start_time
        self.finish_train(train_time)

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
        valid_modes =  {"loss", "acc", "f1", "fpr", "pauc", "last", "all"}
        if not (model_state in valid_modes or isinstance(model_state, int)):
            self.error(f"Invalid model value: {model_state}. Must be one of {valid_modes} or an integer.")

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "f1":
            if self.model_f1 is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_f1
        elif model_state == "fpr":
            if self.model_fpr is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]            


        # Check output_max
        valid_output_types = {"softmax", "argmax", "logits"}
        if output_type not in valid_output_types:
            self.error(f"Invalid output_max value: {output_type}. Must be one of {valid_output_types}.")
        
        y_preds = []
        model.eval()
        model.to(self.device)

        # Set inference context
        try:
            inference_context = torch.inference_mode()
            with torch.inference_mode():
                for batch, (X, y) in enumerate(dataloader):
                    check = self.get_predictions(model(X.to(self.device)))
                    break
        except RuntimeError:
            inference_context = torch.no_grad()
            self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

        # Free up unused GPU memory after shape-checking
        torch.cuda.empty_cache()

        # Attempt a forward pass to check if the shape of X is compatible
        with inference_context:
            for batch, (X, y) in enumerate(dataloader):            
                try:
                    # This is where the model will "complain" if the shape is incorrect
                    check = self.get_predictions(model(X.to(self.device)))
                except Exception as e:
                    # If the shape is wrong, reshape X and try again
                    match = re.search(r"got input of size: (\[[^\]]+\])", str(e))
                    if match:
                        self.warning(f"Wrong input shape: {match.group(1)}. Attempting to reshape X.")
                    else:
                        self.warning(f"Attempting to reshape X.")

                    # Check the current shape of X and attempt a fix
                    if X.ndimension() == 3:  # [batch_size, 1, time_steps]
                        self.squeeze_dim = True
                    elif X.ndimension() == 2:  # [batch_size, time_steps]
                        pass  # No change needed
                    else:
                        raise ValueError(f"Unexpected input shape after exception handling: {X.shape}")
                break
        
        # Free up unused GPU memory after shape-checking
        torch.cuda.empty_cache()      

        # Turn on inference context manager 
        with inference_context:
            for batch, (X, y) in self.progress_bar(
                dataloader=dataloader,
                total=len(dataloader),
                stage='inference'
                ):

                # Send data and targets to target device
                X, y = X.to(self.device), y.to(self.device)
                X = X.squeeze(1) if self.squeeze_dim else X
                
                # Do the forward pass
                y_logit = self.get_predictions(model(X))

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
        transform: Optional[Callable], #Union[torchvision.transforms, torchaudio.transforms], 
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
            transform (Callable): The transformation to apply to the test images.
            class_names (list): A list of class names.
            sample_fraction (float, optional): The fraction of samples to predict. Defaults to 1.0.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            A list of dictionaries with sample predictions, sample names, prediction probabilities, prediction times, actual labels and prediction classes
            A classification report as a dictionary from sckit-learng.metrics
        """

        # Check model to use
        valid_modes =  {"loss", "acc", "f1", "fpr", "pauc", "last", "all"}
        if not (model_state in valid_modes or isinstance(model_state, int)):
            self.error(f"Invalid model value: {model_state}. Must be one of {valid_modes} or an integer.")

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "f1":
            if self.model_f1 is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_f1
        elif model_state == "fpr":
            if self.model_fpr is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]

        # Create a list of test images and checkout existence
        
        # Define valid file extensions
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
        audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a")

        valid_extensions = image_extensions + audio_extensions

        # Collect file paths
        paths = [p for p in Path(test_dir).rglob("*") if p.suffix.lower() in valid_extensions]
        if len(paths) == 0:
            self.error(f"No valid image or audio files found in directory: {test_dir}.")

        # Number of random images to extract
        num_samples = len(paths)
        num_random_samples = int(sample_fraction * num_samples)

        # Ensure the number of images to extract is less than or equal to the total number of images
        if num_random_samples > num_samples:
            self.warning(f"Number of images to extract exceeds total images in directory: {num_samples}. Using all images instead.")
            num_random_samples = num_samples
        #assert num_random_samples <= len(paths), f"Number of images to extract exceeds total images in directory: {len(paths)}"

        # Randomly select a subset of file paths
        torch.manual_seed(seed)
        paths = random.sample(paths, num_random_samples)

        # Store predictions and ground-truth labels
        y_true = []
        y_pred = []

        # Create an empty list to store prediction dictionaires
        pred_list = []
        
        # Loop through target paths
        for path in tqdm(paths, total=num_samples, colour=self.color_test_plt):
            
            # Create empty dictionary to store prediction information for each sample
            pred_dict = {}

            # Get the sample path and ground truth class name
            pred_dict["path"] = path
            class_name = path.parent.stem
            pred_dict["class_name"] = class_name
            
            # Start the prediction timer
            start_time = timer()
            
            # Process image or audio
            if path.suffix.lower() in image_extensions:
                # Load and transform image
                signal = Image.open(path) #.convert("RGB")
            elif path.suffix.lower() in audio_extensions:
                # Load and transform audio
                signal, sample_rate = torchaudio.load(path)
            if transform:
                try:
                    tranform = transform.to(self.device)
                    signal = transform(signal)
                except:
                    # Fall back to cpu if error
                    transform = transform.to("cpu")
                    signal = transform(signal)
                
            signal = signal.unsqueeze(0).to(self.device)

            # Prepare model for inference by sending it to target device and turning on eval() mode
            model.to(self.device)
            model.eval()

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():        
                    check = model(signal)
            except RuntimeError:
                inference_context = torch.no_grad()
                self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")
            
            # Attempt a forward pass to check if the shape of transformed_image is compatible
            try:
                # This is where the model will "complain" if the shape is incorrect
                check = model(signal)
            except Exception as e:
                # If the shape is wrong, reshape X and try again
                match = re.search(r"got input of size: (\[[^\]]+\])", str(e))
                if match:
                    self.warning(f"Wrong input shape: {match.group(1)}. Attempting to reshape X.")
                else:
                    self.warning(f"Attempting to reshape X.")

                # Check the current shape of X and attempt a fix
                if signal.ndimension() == 3:  # [batch_size, 1, time_steps]
                    self.squeeze_dim = True
                elif signal.ndimension() == 2:  # [batch_size, time_steps]
                    pass  # No change needed
                else:
                    raise ValueError(f"Unexpected input shape after exception handling: {signal.shape}")
            
            # Get prediction probability, predicition label and prediction class
            with inference_context:
                signal = signal.squeeze(1) if self.squeeze_dim else signal
                pred_logit = model(signal) # perform inference on target sample 
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

            clear_output(wait=True)

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
class DistillationEngine(Common):

    """
    A class to handle training, evaluation, and predictions for a PyTorch model.

    This class provides functionality to manage the training and evaluation lifecycle
    of a PyTorch model, including saving the best model based on specific criteria.

    Args:
        student (torch.nn.Module, optional): The PyTorch model for the student to handle. Must be instantiated.
        teacher (torch.nn.Module, optional): The PyTorch model for the teacher to handle. Must be instantiated.
        color_map (dict, optional): Specifies the colors for the training and evaluation curves:
        'black', 'blue', 'orange', 'green', 'red', 'yellow', 'magenta', 'cyan', 'white',
        'light_gray', 'dark_gray', 'light_blue', 'light_green', 'light_red', 'light_yellow',
        'light_magenta', 'light_cyan'.
        log_verbose (boo, optional): if True, activate logger messages.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(
        self,
        student: torch.nn.Module=None,
        teacher: torch.nn.Module=None,
        color_map: dict=None,
        log_verbose: bool=True,
        device: str="cuda" if torch.cuda.is_available() else "cpu"
        ):

        super().__init__()

        # Initialize self variables
        self.device = device
        self.model = student
        self.model_tch = teacher
        self.model_acc = None
        self.model_f1 = None
        self.model_loss = None
        self.model_fpr = None
        self.model_pauc = None
        self.model_epoch = None
        self.save_best_model = False
        self.keep_best_models_in_memory = False
        self.mode = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.model_name = None
        self.model_name_loss = None
        self.model_name_acc = None
        self.model_name_f1 = None
        self.model_name_fpr = None
        self.model_name_pauc = None
        self.log_verbose = log_verbose
     
        # Initialize colors
        default_color_map = {'train': 'blue', 'test': 'orange', 'other': 'black'}
        
        # If the user provides a color_map, update the default with it
        if color_map is None:
            color_map = default_color_map # Use defaults if no user input
        else:
            color_map = {**default_color_map, **color_map} # Merge user input with defaults

        self.color_train = Colors.get_console_color(color_map['train'])
        self.color_test =  Colors.get_console_color(color_map['test'])
        self.color_other = Colors.get_console_color(color_map['other'])
        self.color_train_plt = Colors.get_matplotlib_color(color_map['train'])
        self.color_test_plt =  Colors.get_matplotlib_color(color_map['test'])

        # Initialize result logs
        self.results = {}

        # Check if model is provided
        if self.model is None:
            self.error(f"Instantiate the engine by passing a PyTorch model to handle.")
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
        if not any(model_name.endswith(ext) for ext in valid_extensions):
            self.error(f"'model_name' should end with one of {valid_extensions}.")
        #assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
        model_save_path = Path(target_dir) / model_name

        # Save the model state_dict()
        self.info(f"Saving model to: {model_save_path}")
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

        # Define the list of valid extensions
        valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

        # Create the model path
        if not any(model_name.endswith(ext) for ext in valid_extensions):
            self.error(f"'model_name' should end with one of {valid_extensions}.")
        #assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_path = Path(target_dir) / model_name

        # Load the model
        self.info(f"Loading model from: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        
        return self
    
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
            ):
        
        """
        Prints the configuration of the training process.
        """

        self.info(f"Device: {self.device}")
        self.info(f"Epochs: {epochs}")
        self.info(f"Batch size: {batch_size}")
        self.info(f"Accumulation steps: {accumulation_steps}")
        self.info(f"Effective batch size: {batch_size * accumulation_steps}")
        self.info(f"Recall threshold - fpr: {recall_threshold}")
        self.info(f"Recall threshold - pauc: {recall_threshold_pauc}")
        self.info(f"Apply validation: {self.apply_validation}")
        self.info(f"Plot curves: {plot_curves}")
        self.info(f"Automatic Mixed Precision (AMP): {amp}")
        self.info(f"Enable clipping: {enable_clipping}")
        self.info(f"Debug mode: {debug_mode}")
        self.info(f"Save model: {self.save_best_model}")
        self.info(f"Target directory: {self.target_dir}")        
        if self.save_best_model:
          # Extract base name and extension from the model name
            base_name, extension = os.path.splitext(self.model_name)
            
            # Print base name and extension
            self.info(f"Model name base: {base_name}")
            self.info(f"Model name extension: {extension}")
            
            # Iterate over modes and format model name, skipping 'last'
            for mode in self.mode:
                if mode == "last" or mode == "all":
                    # Skip adding 'last' and just use epoch in the filename
                    model_name_with_mode = f"_epoch<int>{extension}"
                else:
                    # For other modes, include mode and epoch in the filename
                    model_name_with_mode = f"_{mode}_epoch<int>{extension}"
                
                # Print the final model save path for each mode
                self.info(f"Save best model - {mode}: {base_name + model_name_with_mode}")
        if self.keep_best_models_in_memory:
            self.warning(f"Keeping best models in memory: {self.keep_best_models_in_memory} - it may slow down the training process.")
        else:
            self.info(f"Keeping best models in memory: {self.keep_best_models_in_memory}")

    # Function to initialize the result logs
    def init_results(self):

        """Creates a empty results dictionary"""

        self.results = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "train_fpr": [],
            "train_pauc": [],
            "train_time [s]": [],
            "test_loss": [],
            "test_acc": [],
            "test_f1": [],
            "test_fpr": [],
            "test_pauc": [],
            "test_time [s]": [],
            "lr": [],
            }
    
    # Function to display and plot the results
    def display_results(
        self,
        epoch,
        max_epochs,
        train_loss,
        train_acc,
        train_f1,
        recall_threshold,
        recall_threshold_pauc,
        train_fpr,
        train_pauc,
        train_epoch_time,
        test_loss,
        test_acc,
        test_f1,
        test_fpr,
        test_pauc,
        test_epoch_time,
        plot_curves,
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
            f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
            f"{self.color_train}Train: {self.color_other}| "
            f"{self.color_train}loss: {train_loss:.4f} {self.color_other}| "
            f"{self.color_train}acc: {train_acc:.4f} {self.color_other}| "
            f"{self.color_train}f1: {train_f1:.4f} {self.color_other}| "
            f"{self.color_train}fpr: {train_fpr:.4f} {self.color_other}| "
            f"{self.color_train}pauc: {train_pauc:.4f} {self.color_other}| "
            f"{self.color_train}time: {self.sec_to_min_sec(train_epoch_time)} {self.color_other}| "            
            f"{self.color_train}lr: {lr:.10f}"
        )
        if self.apply_validation:
            print(
                f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
                f"{self.color_test}Test:  {self.color_other}| "
                f"{self.color_test}loss: {test_loss:.4f} {self.color_other}| "
                f"{self.color_test}acc: {test_acc:.4f} {self.color_other}| "
                f"{self.color_test}f1: {test_f1:.4f} {self.color_other}| "
                f"{self.color_test}fpr: {test_fpr:.4f} {self.color_other}| "
                f"{self.color_test}pauc: {test_pauc:.4f} {self.color_other}| "
                f"{self.color_test}time: {self.sec_to_min_sec(test_epoch_time)} {self.color_other}| "            
                f"{self.color_test}lr: {lr:.10f}"
            )
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)
        self.results["train_loss"].append(train_loss)
        self.results["train_acc"].append(train_acc)
        self.results["train_f1"].append(train_f1)
        self.results["test_loss"].append(test_loss)
        self.results["test_acc"].append(test_acc)
        self.results["test_f1"].append(test_f1)
        self.results["train_time [s]"].append(train_epoch_time)
        self.results["test_time [s]"].append(test_epoch_time)
        self.results["lr"].append(lr)
        self.results["train_fpr"].append(train_fpr)
        self.results["test_fpr"].append(test_fpr)
        self.results["train_pauc"].append(train_pauc)
        self.results["test_pauc"].append(test_pauc)

        # Plots training and test loss, accuracy, and fpr-at-recall curves.
        if plot_curves:
        
            n_plots = 5
            plt.figure(figsize=(25, 6))
            range_epochs = range(1, len(self.results["train_loss"])+1)

            # Plot loss
            plt.subplot(1, n_plots, 1)
            plt.plot(range_epochs, self.results["train_loss"], label="train_loss", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_loss"], label="test_loss", color=self.color_test_plt)
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot accuracy
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, self.results["train_acc"], label="train_accuracy", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_acc"], label="test_accuracy", color=self.color_test_plt)
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot F1-score
            plt.subplot(1, n_plots, 3)
            plt.plot(range_epochs, self.results["train_f1"], label="train_f1", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_f1"], label="test_f1", color=self.color_test_plt)
            plt.title("F1-Score")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            # Plot FPR at recall
            plt.subplot(1, n_plots, 4)
            plt.plot(range_epochs, self.results["train_fpr"], label="train_fpr_at_recall", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_fpr"], label="test_fpr_at_recall", color=self.color_test_plt)
            plt.title(f"FPR at {recall_threshold * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot pAUC at recall
            plt.subplot(1, n_plots, 5)
            plt.plot(range_epochs, self.results["train_pauc"], label="train_pauc_at_recall", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_pauc"], label="test_pauc_at_recall", color=self.color_test_plt)
            plt.title(f"pAUC above {recall_threshold_pauc * 100}% recall")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
                    
            plt.show()

    def init_train(
        self,
        target_dir: str=None,
        model_name: str=None,
        dataloader: torch.utils.data.DataLoader=None,
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
        debug_mode: bool=False
        ):

        """
        Initializes the training process by setting up the required configurations, parameters, and resources.

        Args:
            target_dir (str, optional): Directory to save the models. Defaults to "models" if not provided.
            model_name (str, optional): Name of the model file to save. Defaults to the class name of the model with ".pth" extension.
            dataloader: A DataLoader instance for the model to be trained on.
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
            self.error(f"'keep_best_models_in_memory' must be True or False.")
        else:
            self.keep_best_models_in_memory = keep_best_models_in_memory

        # Validate apply_validation
        if not isinstance(apply_validation, (bool)):
            self.error(f"'apply_validation' must be True or False.")
        else:
            self.apply_validation = apply_validation
      
        # Validate recall_threshold
        if not isinstance(recall_threshold, (int, float)) or not (0.0 <= float(recall_threshold) <= 1.0):
            self.error(f"'recall_threshold' must be a float between 0.0 and 1.0.")

        # Validate recall_threshold_pauc
        if not isinstance(recall_threshold_pauc, (int, float)) or not (0.0 <= float(recall_threshold_pauc) <= 1.0):
            self.error(f"'recall_threshold_pauc' must be a float between 0.0 and 1.0.")

        # Validate accumulation_steps
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            self.error(f"'accumulation_steps' must be an integer greater than or equal to 1.")

        # Validate epochs
        if not isinstance(epochs, int) or epochs < 1:
            self.error(f"'epochs' must be an integer greater than or equal to 1.")

        # Ensure save_best_model is correctly handled
        if save_best_model is None:
            self.save_best_model = False
            mode = []
        elif isinstance(save_best_model, (str, list)):
            self.save_best_model = True
            mode = [save_best_model] if isinstance(save_best_model, str) else save_best_model  # Ensure mode is a list
        else:
            self.error(f"'save_best_model' must be None, a string, or a list of strings.")

        # Validate mode only if save_best_model is True
        valid_modes = {"loss", "acc", "f1", "fpr", "pauc", "last", "all"}
        if self.save_best_model:
            if not isinstance(mode, list):
                self.error(f"'mode' must be a string or a list of strings.")

            for m in mode:
                if m not in valid_modes:
                    self.error(f"Invalid mode value: '{m}'. Must be one of {valid_modes}")

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
            debug_mode=debug_mode
            )
        
        # Initialize optimizer, loss_fn, and scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.init_results()

        # Attempt a forward pass to check if the shape of X is compatible
        for batch, (X, y) in enumerate(dataloader):
            try:
                # This is where the model will "complain" if the shape is incorrect
                check = self.get_predictions(self.model(X.to(self.device)))
            except Exception as e:
                # If the shape is wrong, reshape X and try again
                match = re.search(r"got input of size: (\[[^\]]+\])", str(e))
                if match:
                    self.warning(f"Wrong input shape: {match.group(1)}. Attempting to reshape X.")
                else:
                    self.warning(f"Attempting to reshape X.")

                # Check the current shape of X and attempt a fix
                if X.ndimension() == 3:  # [batch_size, 1, time_steps]
                    self.squeeze_dim = True
                elif X.ndimension() == 2:  # [batch_size, time_steps]
                    pass  # No change needed
                else:
                    self.error(f"Unexpected input shape after exception handling: {X.shape}")
            break
    
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
            if "f1" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_f1 = copy.deepcopy(self.model)                            
                    self.model_f1.to(self.device)
                self.model_name_f1 = self.model_name.replace(".", f"_f1.")
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
            self.best_test_f1 = 0.0
            self.best_test_fpr = float("inf")
            self.best_test_pauc = 0.0
    
    def progress_bar(
        self,
        dataloader: torch.utils.data.DataLoader,
        total: int,
        stage: str,
        epoch_number: int = 1,
        desc_length: int = 22):

        """
        Creates the tqdm progress bar for the training and validation stages.

        Args:
            dataloader: The dataloader for the current stage.
            total: The total number of batches in the dataloader.
            stage: The current stage ("train" or "validate").
            epoch_number: The current epoch number.
            desc_length: The length of the description string for the progress bar.

        Returns:
            A tqdm progress bar instance for the current stage.
        """

        train_str = f"Training epoch {epoch_number+1}"
        val_str = f"Validating epoch {epoch_number+1}"
        
        if stage == 'train':
            color = self.color_train_plt
            desc = f"Training epoch {epoch_number+1}".ljust(desc_length) + " "
        elif stage == 'validation' or stage == "test":
            color = self.color_test_plt
            desc = f"Validating epoch {epoch_number+1}".ljust(desc_length) + " "
        else:
            color = self.color_test_plt
            desc = f"Making predictions"
        progress = tqdm(enumerate(dataloader), total=total, colour=color)
        progress.set_description(desc)

        return progress

    # This train step function includes gradient accumulation (experimental)
    def train_step_v2(
        self,
        dataloader_std: torch.utils.data.DataLoader, 
        dataloader_tch: torch.utils.data.DataLoader,
        num_classes: int=2,
        recall_threshold: float=0.99,
        recall_threshold_pauc: float=0.0,
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
            In the form (train_loss, train_acc, train_f1, train_fpr, train_pauc). For example: (0.1112, 0.8743, 0.18332, 0.01123, 0.15561).
        """

        # Put student model in train mode
        self.model.train()
        self.model.to(self.device)

        # Put teacher model in evaluation mode
        self.model_tch.eval()
        self.model_tch.to(self.device)

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        len_dataloader = len(dataloader_std)
        train_loss, train_acc, train_f1 = 0, 0, 0
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting        
        for (batch, (X, y)), (_, (X_tch, _)) in self.progress_bar(
            dataloader_std=dataloader_std,
            dataloader_tch=dataloader_tch,
            total=len_dataloader,
            epoch_number=epoch_number,
            stage="train"):
            
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
                            self.warning(f"y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            self.warning(f"y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue
                    
                    # Calculate loss, normalize by accumulation steps
                    loss = self.loss_fn(y_pred, y_pred_tch, y) / accumulation_steps
                
                    # Check for NaN or Inf in loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        self.warning(f"Loss is NaN or Inf at batch {batch}. Skipping...")
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
            if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len_dataloader:

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
                                    self.warning(f"NaN or Inf gradient detected in {name} at batch {batch}")
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
            train_acc += self.calculate_accuracy(y, y_pred_class)

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
            train_f1 = self.calculate_f1_score(all_labels, all_preds.argmax(dim=1), num_classes)
        except Exception as e:
            self.warning(f"Innacurate calculation of F1-score: {e}")
        try:    
            train_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
        except Exception as e:
            self.error(f"Innacurate calculation of final FPR at recall: {e}")
            train_fpr = 1.0
        try:    
            train_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
        except Exception as e:
            self.error(f"Innacurate calculation of final pAUC at recall: {e}")
            train_pauc = 0.0

        return train_loss, train_acc, train_f1, train_fpr, train_pauc

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

            # Put the student model in eval mode
            self.model.eval() 
            self.model.to(self.device)

            # Put the teacher model in eval mode
            self.model_tch.eval()
            self.model_tch.to(self.device)

            # Setup test loss and test accuracy values
            len_dataloader = len(dataloader_std)
            test_loss, test_acc, test_f1 = 0, 0, 0
            all_preds = []
            all_labels = []

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():        
                    for batch, (X, y) in enumerate(dataloader_std):
                        test_pred = self.get_predictions(self.model(X.to(self.device)))
                        break
            except RuntimeError:
                inference_context = torch.no_grad()
                self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

            # Turn on inference context manager 
            with inference_context:

                # Loop through DataLoader batches                   
                for (batch, (X, y)), (_, (X_tch, _)) in self.progress_bar(
                    dataloader_std=dataloader_std,
                    dataloader_tch=dataloader_tch,
                    total=len_dataloader,
                    epoch_number=epoch_number,
                    stage="test"
                    ):

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
                                self.warning(f"Predictions contain NaN/Inf at batch {batch}. Applying clipping...")
                                test_pred = torch.nan_to_num(
                                    test_pred,
                                    nan=torch.mean(test_pred).item(),
                                    posinf=torch.max(test_pred).item(),
                                    neginf=torch.min(test_pred).item()
                                )
                            else:
                                self.warning(f"Predictions contain NaN/Inf at batch {batch}. Skipping batch...")
                                continue

                        # Calculate and accumulate loss
                        loss = self.loss_fn(test_pred, test_pred_tch, y)
                        test_loss += loss.item()

                        # Debug NaN/Inf loss
                        if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                            self.warning(f"Loss is NaN/Inf at batch {batch}. Skipping...")
                            continue

                    # Calculate and accumulate accuracy
                    test_pred = test_pred.float() # Convert to float for stability
                    test_pred_class = test_pred.argmax(dim=1)
                    test_acc += self.calculate_accuracy(y, test_pred_class)

                    # Collect outputs for fpr-at-recall calculation
                    all_preds.append(torch.softmax(test_pred, dim=1).detach().cpu())
                    all_labels.append(y.detach().cpu())

            # Adjust metrics to get average loss and accuracy per batch 
            test_loss = test_loss / len_dataloader
            test_acc = test_acc / len_dataloader            

            # Final FPR calculation
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            try:
                test_f1 = self.calculate_f1_score(all_labels, all_preds.argmax(dim=1), num_classes)
            except Exception as e:
                self.warning(f"Innacurate calculation of F1-score: {e}")
            try:    
                test_fpr = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)            
            except Exception as e:
                self.error(f"Innacurate calculation of final FPR at recall: {e}")
                test_fpr = 1.0
            try:    
                test_pauc = self.calculate_pauc_at_recall(all_labels, all_preds, recall_threshold_pauc, num_classes)
            except Exception as e:
                self.error(f"Innacurate calculation of final pAUC at recall: {e}")
                test_pauc = 0.0
        
        # Otherwise set params with initial values
        else:
            test_loss, test_acc, test_f1, test_fpr, test_pauc = self.best_test_loss, self.best_test_acc, self.best_test_f1, self.best_test_fpr, self.best_test_pauc

        return test_loss, test_acc, test_f1, test_fpr, test_pauc

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
                    self.error(
                        f"The scheduler requires either `test_loss` or `test_acc` "
                        "depending on its mode ('min' or 'max')."
                        )
            else:
                self.scheduler.step()  # For other schedulers
    
    # Updates and saves the best model and model_epoch list based on the specified mode.
    def update_model(
        self,
        test_loss: float = None,
        test_acc: float = None,
        test_f1: float = None,
        test_fpr: float = None,
        test_pauc: float = None,
        epoch: int = None,
    ) -> pd.DataFrame:

        """
        Updates and saves the best model and model_epoch list based on the specified mode(s), as well as the last-epoch model.

        Parameters:
        - test_loss (float, optional): Test loss for the current epoch (used in "loss" mode).
        - test_acc (float, optional): Test accuracy for the current epoch (used in "acc" mode).
        - test_f1 (float, optional): Test F1-score for the current epoch (used in "f1" mode).
        - test_fpr (float, optional): Test false positive rate at the specified recall (used in "fpr" mode).
        - test_pauc (float, optional): Test pAUC at the specified recall (used in "pauc" mode).
        - epoch (int, optional): Current epoch index, used for naming models when saving all epochs in "all" mode.

        Functionality:
        - Saves the last-epoch model.
        - Saves the logs (self.results).
        - Saves the best-performing model during training based on the specified evaluation mode.
        - If mode is "all", saves the model for every epoch.

        Returns:
            A dataframe of training and testing metrics for each epoch.
        """

        if isinstance(self.mode, str):
            self.mode = [self.mode]  # Ensure self.mode is always a list

        if epoch is None:
            self.error(f"'epoch' must be provided when mode includes 'all' or 'last'.")

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
                        self.error(f"'test_loss' must be provided when mode is 'loss'.")
                    if test_loss < self.best_test_loss:
                        remove_previous_best(self.model_name_loss)
                        self.best_test_loss = test_loss
                        if self.keep_best_models_in_memory:
                            self.model_loss.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_loss)
                # Accuracy criterion    
                elif mode == "acc":
                    if test_acc is None:
                        self.error(f"'test_acc' must be provided when mode is 'acc'.")
                    if test_acc > self.best_test_acc:
                        remove_previous_best(self.model_name_acc)
                        self.best_test_acc = test_acc
                        if self.keep_best_models_in_memory:
                            self.model_acc.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_acc)
                # F1-score criterion    
                elif mode == "f1":
                    if test_f1 is None:
                        self.error(f"'test_f1' must be provided when mode is 'acc'.")
                    if test_f1 > self.best_test_f1:
                        remove_previous_best(self.model_name_acc)
                        self.best_test_f1 = test_f1
                        if self.keep_best_models_in_memory:
                            self.model_f1.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_f1)
                # FPR criterion
                elif mode == "fpr":
                    if test_fpr is None:
                        self.error(f"'test_fpr' must be provided when mode is 'fpr'.")
                    if test_fpr < self.best_test_fpr:
                        remove_previous_best(self.model_name_fpr)
                        self.best_test_fpr = test_fpr
                        if self.keep_best_models_in_memory:
                            self.model_fpr.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_fpr)
                # pAUC criterion    
                elif mode == "pauc":
                    if test_pauc is None:
                        self.error(f"'test_pauc' must be provided when mode is 'pauc'.")
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
        train_time: float=None
        ):

        """
        Finalizes the training process by showing the elapsed time.
        
        Args:
            train_time: Elapsed time.
        """

        # Print elapsed time
        self.info(f"Training finished! Elapsed time: {self.sec_to_min_sec(train_time)}")

    
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
        recall_threshold: float=0.99,
        recall_threshold_pauc: float=0.0,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=False,
        accumulation_steps: int=1,
        debug_mode: bool=False
        ) -> pd.DataFrame:

            
        """Trains and tests a PyTorch model using both student and teacher models.

        The function passes the student and teacher models through the train_step() 
        and test_step() functions for a number of epochs, training and testing the 
        models in the same epoch loop.

        Calculates, prints, and stores evaluation metrics throughout.

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

        Returns:
            A dataframe of training and testing loss, accuracy, FPR at recall, and pAUC 
            for each epoch. The dataframe contains the following columns:
                - epoch: The epoch number.
                - train_loss: The training loss for each epoch.
                - train_acc: The training accuracy for each epoch.
                - train_f1: The training F1-score for each epoch.
                - test_loss: The testing loss for each epoch.
                - test_acc: The testing accuracy for each epoch.
                - test_f1: The testing F1-score for each epoch.
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
                    train_f1: [0.4258, 0.4712],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973],
                    test_f1: [0.2834, 0.3448],
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
            dataloader=train_dataloader_std,
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
            debug_mode=debug_mode
            )

        # Loop through training and testing steps for a number of epochs
        for epoch in range(epochs):

            # Perform training step and time it
            train_epoch_start_time = time.time()
            train_loss, train_acc, train_f1, train_fpr, train_pauc = self.train_step_v2(
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
            test_loss, test_acc, test_f1, test_fpr, test_pauc = self.test_step(
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
                train_f1=train_f1,
                recall_threshold=recall_threshold,
                recall_threshold_pauc=recall_threshold_pauc,
                train_fpr=train_fpr,
                train_pauc=train_pauc,
                train_epoch_time=train_epoch_time,
                test_loss=test_loss,
                test_acc=test_acc,
                test_f1=test_f1,
                test_fpr=test_fpr,
                test_pauc=test_pauc,
                test_epoch_time=test_epoch_time,
                plot_curves=plot_curves
            )

            # Scheduler step after the optimizer
            self.scheduler_step(
                test_loss=test_loss,
                test_acc=test_acc,
            )

            # Update and save the best model, model_epoch list based on the specified mode, and the actual-epoch model.
            # If apply_validation is enabled then upate models based on validation results
            df_results = self.update_model(
                test_loss=test_loss if self.apply_validation else train_loss,
                test_acc=test_acc if self.apply_validation else train_acc,
                test_f1=test_f1 if self.apply_validation else train_f1,                
                test_fpr=test_fpr if self.apply_validation else train_fpr,
                test_pauc=test_pauc if self.apply_validation else train_pauc,
                epoch=epoch
                )

        # Finish training process
        train_time = time.time() - train_start_time
        self.finish_train(train_time)

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
        valid_modes =  {"loss", "acc", "f1", "fpr", "pauc", "last", "all"}
        if model_state not in valid_modes or not isinstance(model_state, int):
            self.error(f"Invalid model value: {model_state}. Must be one of {valid_modes} or an integer.")
        #assert model_state in valid_modes or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_modes} or an integer."

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "f1":
            if self.model_f1 is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_f1
        elif model_state == "fpr":
            if self.model_fpr is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]            


        # Check output_max
        valid_output_types = {"softmax", "argmax", "logits"}
        if output_type not in valid_output_types:
            self.error(f"Invalid output_max value: {output_type}. Must be one of {valid_output_types}.")
        #assert output_type in valid_output_types, f"Invalid output_max value: {output_type}. Must be one of {valid_output_types}"

        y_preds = []
        model.eval()
        model.to(self.device)

        # Set inference context
        try:
            inference_context = torch.inference_mode()
            with torch.inference_mode():        
                for batch, (X, y) in enumerate(dataloader):
                    test_pred = self.get_predictions(self.model(X.to(self.device)))
                    break
        except RuntimeError:
            inference_context = torch.no_grad()
            self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

        # Turn on inference context manager 
        with inference_context:
            for batch, (X, y) in self.progress_bar(
                dataloader=dataloader,
                total=len(dataloader),
                stage='inference'
                ):

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
        transform: Optional[Callable], #torchvision.transforms, 
        class_names: List[str], 
        model_state: str="last",
        sample_fraction: float=1.0,
        seed=42,        
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        
        """
        Predicts classes for a given dataset using a trained model and stores the per-sample results in dictionaries.

        Args:
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
            test_dir (str): The directory containing the test data.
            transform (Callable): The transformation to apply to the test images.
            class_names (list): A list of class names.
            sample_fraction (float, optional): The fraction of samples to predict. Defaults to 1.0.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            A list of dictionaries with sample predictions, sample names, prediction probabilities, prediction times, actual labels and prediction classes
            A classification report as a dictionary from sckit-learng.metrics
        """

        # Check model to use
        valid_modes =  {"loss", "acc", "f1", "fpr", "pauc", "last", "all"}
        if model_state not in valid_modes or not isinstance(model_state, int):
            self.error(f"Invalid model value: {model_state}. Must be one of {valid_modes} or an integer.")
        #assert model_state in valid_modes or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_modes} or an integer."

        if model_state == "last":
            model = self.model
        elif model_state == "loss":
            if self.model_loss is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_loss
        elif model_state == "acc":
            if self.model_acc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_acc
        elif model_state == "f1":
            if self.model_f1 is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_f1
        elif model_state == "fpr":
            if self.model_fpr is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_fpr
        elif model_state == "pauc":
            if self.model_pauc is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_pauc
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]

        # Create a list of test images and checkout existence
        #file_type: str='jpg',
        #self.info(f"Finding all filepaths ending with '.{file_type}' in directory: {test_dir}")
        #paths = list(Path(test_dir).glob(f"*/*.{file_type}"))
        #assert len(list(paths)) > 0, f"No files ending with '.{file_type}' found in this directory: {test_dir}"

        # Define valid file extensions
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
        audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a")

        valid_extensions = image_extensions + audio_extensions

        # Collect file paths
        paths = [p for p in Path(test_dir).rglob("*") if p.suffix.lower() in valid_extensions]
        if len(paths) == 0:
            self.error(f"No valid image or audio files found in directory: {test_dir}.")
        #assert len(paths) > 0, f"No valid image or audio files found in directory: {test_dir}"

        # Number of random images to extract
        num_samples = len(paths)
        num_random_samples = int(sample_fraction * num_samples)

        # Ensure the number of images to extract is less than or equal to the total number of images
        if num_random_samples > num_samples:
            self.warning(f"Number of images to extract exceeds total images in directory: {num_samples}. Using all images instead.")
            num_random_samples = num_samples
        #assert num_random_samples <= len(paths), f"Number of images to extract exceeds total images in directory: {len(paths)}"

        # Randomly select a subset of file paths
        torch.manual_seed(seed)
        paths = random.sample(paths, num_random_samples)

        # Store predictions and ground-truth labels
        y_true = []
        y_pred = []

        # Create an empty list to store prediction dictionaires
        pred_list = []
        
        # Loop through target paths
        for path in tqdm(paths, total=num_samples, colour=self.color_test_plt):
            
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

            # Attempt a forward pass to check if the shape of transformed_image is compatible
            try:
                # This is where the model will "complain" if the shape is incorrect
                check = model(transformed_image)
            except Exception as e:
                # If the shape is wrong, reshape X and try again
                match = re.search(r"got input of size: (\[[^\]]+\])", str(e))
                if match:
                    self.warning(f"Wrong input shape: {match.group(1)}. Attempting to reshape X.")
                else:
                    self.warning(f"Attempting to reshape X.")

                # Check the current shape of X and attempt a fix
                if transformed_image.ndimension() == 3:  # [batch_size, 1, time_steps]
                    self.squeeze_dim = True
                elif transformed_image.ndimension() == 2:  # [batch_size, time_steps]
                    pass  # No change needed
                else:
                    raise ValueError(f"Unexpected input shape after exception handling: {transformed_image.shape}")

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():        
                    check = model(transformed_image)
            except RuntimeError:
                inference_context = torch.no_grad()
                self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")
            
            # Get prediction probability, predicition label and prediction class
            with inference_context:
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