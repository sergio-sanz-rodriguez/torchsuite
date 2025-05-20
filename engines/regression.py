"""
Contains classes for training and testing a PyTorch model for classification tasks.  
"""

import os
import glob
import torch
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
import seaborn as sns
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report
from contextlib import nullcontext
from .common import Common, Colors

import warnings
warnings.filterwarnings("ignore")

            
# Training and prediction engine class
class RegressionEngine(Common):

    """
    A class to handle training, evaluation, and predictions for a PyTorch model.

    This class provides functionality to manage the training and evaluation lifecycle
    of a PyTorch model, including saving the best model based on specific criteria.

    Args:
        - model (torch.nn.Module, optional): The PyTorch model to handle. Must be instantiated.
        - color_map (dict, optional): Specifies the colors for the training and evaluation curves:
          'black', 'blue', 'orange', 'green', 'red', 'yellow', 'magenta', 'cyan', 'white',
          'light_gray', 'dark_gray', 'light_blue', 'light_green', 'light_red', 'light_yellow',
          'light_magenta', 'light_cyan'.
          Example: {'train': 'blue', 'test': 'orange', 'other': 'black'}
        - log_verbose (bool, optional): if True, activate logger messages.
        - device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
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
        self.model_loss = None     
        self.model_r2 = None   
        self.model_epoch = None
        self.save_best_model = False
        self.keep_best_models_in_memory = False
        self.mode = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.model_name = None
        self.model_name_loss = None        
        self.model_name_r2 = None
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
        self.color_other_plt =  Colors.get_matplotlib_color(color_map['other'])
        

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
        self.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")        
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
                if mode == "last" or mode == "all" or mode == "r2":
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
            "train_r2": [],
            "train_time [s]": [],
            "test_loss": [],            
            "test_r2": [],
            "test_time [s]": [],
            "lr": [],
            }
    
    # Function to display and plot the results
    def display_results(
        self,
        epoch,
        max_epochs,
        train_loss,
        train_r2,    
        train_pred,
        train_y,           
        train_epoch_time,
        test_loss,        
        test_r2,
        test_pred, 
        test_y,               
        test_epoch_time,
        plot_curves,
        ):
    
        """
        Displays the training and validation results both numerically and visually.

        Functionality:
        - Outputs key metrics such as training and validation loss, accuracy, and fpr at recall in numerical form.
        - Generates plots that visualize the training process, such as:
        - Loss curves (training vs validation loss over epochs).
        - R2 curves (training vs validation accuracy over epochs).
        - Data distribtutions
        - LR curve
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
            f"{self.color_train}r2: {train_r2:.4f} {self.color_other}| "            
            f"{self.color_train}time: {self.sec_to_min_sec(train_epoch_time)} {self.color_other}| "            
            f"{self.color_train}lr: {lr:.10f}"
        )
        if self.apply_validation:
            print(
                f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
                f"{self.color_test}Test:  {self.color_other}| "
                f"{self.color_test}loss: {test_loss:.4f} {self.color_other}| "
                f"{self.color_test}r2: {test_r2:.4f} {self.color_other}| "            
                f"{self.color_test}time: {self.sec_to_min_sec(test_epoch_time)} {self.color_other}| "            
                f"{self.color_test}lr: {lr:.10f}"
            )
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)
        self.results["train_loss"].append(train_loss)        
        self.results["test_loss"].append(test_loss)
        self.results["train_r2"].append(train_r2)        
        self.results["test_r2"].append(test_r2)
        self.results["train_time [s]"].append(train_epoch_time)
        self.results["test_time [s]"].append(test_epoch_time)
        self.results["lr"].append(lr)        

        # Plots training and test loss, accuracy, and fpr-at-recall curves.
        if plot_curves:
        
            n_plots = 4
            plt.figure(figsize=(20, 6))
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

            # Plot R2
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, self.results["train_r2"], label="train_r2", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_r2"], label="test_r2", color=self.color_test_plt)
            plt.title("R2")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            plt.subplot(1, n_plots, 3)            
            sns.kdeplot(train_y, label="train_kde_gt", color=self.color_other_plt, fill=True)
            sns.kdeplot(train_pred, label="train_kde_pred", color=self.color_train_plt, fill=True)
            plt.title('Prediction Score Distributions')
            plt.xlabel('Score')
            plt.ylabel('Density')
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            plt.subplot(1, n_plots, 4)            
            sns.kdeplot(test_y, label="test_kde_gt", color=self.color_other_plt, fill=True)
            sns.kdeplot(test_pred, label="test_kde_pred", color=self.color_test_plt, fill=True)
            plt.title('Prediction Score Distributions')
            plt.xlabel('Score')
            plt.ylabel('Density')
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot LR
            #plt.subplot(1, n_plots, 3)
            #plt.plot(range_epochs, self.results["lr"], label="lr", color=self.color_train_plt)            
            #plt.title("Learning Rate")
            #plt.xlabel("Epochs")
            #plt.grid(visible=True, which="both", axis="both")
            #plt.legend()

            #png_name = os.path.splitext(self.model_name)[0]
            #png_name = png_name + ".png"
            #plt.savefig(png_name)

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
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=False,
        accumulation_steps: int=1,
        debug_mode: bool=False,
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
            epochs (int, optional): Number of epochs to train. Must be an integer greater than or equal to 1. Default is 30.
            plot_curves (bool, optional): Whether to plot training and validation curves. Default is True.
            amp (bool, optional): Enable automatic mixed precision for faster training. Default is True.
            enable_clipping (bool, optional): Whether to enable gradient clipping. Default is True.
            accumulation_steps (int, optional): Steps for gradient accumulation. Must be an integer greater than or equal to 1. Default is 1.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.
            save_best_model (Union[str, List[str]]): Criterion mode for saving the model: 
                - "loss": saves the epoch with the lowest validation loss                
                - "r2": saves the epoch with the highest r2 value
                - "last": saves last epoch
                - "all": saves models for all epochs
                - A list, e.g., ["loss", "last"], is also allowed. Only applicable if `save_best_model` is True.
                - None: the model will not be saved.

        Functionality:
            Validates `epochs` parameters with assertions.
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
        valid_modes = {"loss", "r2", "last", "all"}
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

        # Initialize optimizer, loss_fn, scheduler, and result_log
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.scaler = GradScaler() if amp else None
        self.init_results()
                
        # Print configuration parameters
        self.print_config(
            batch_size=batch_size,            
            epochs=epochs,
            plot_curves=plot_curves,
            amp=amp,
            enable_clipping=enable_clipping,
            accumulation_steps=accumulation_steps,            
            debug_mode=debug_mode
            )
        
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
                if X.ndimension() == 3 and X.shape[1] == 1:  # [batch_size, 1, time_steps]
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
            if "r2" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_r2 = copy.deepcopy(self.model)                            
                    self.model_r2.to(self.device)
                self.model_name_r2 = self.model_name.replace(".", f"_r2.")            
            if "all" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_epoch = []
                    for k in range(epochs):
                        self.model_epoch.append(copy.deepcopy(self.model))
                        self.model_epoch[k].to(self.device)
            self.best_test_loss = float("inf")             
            self.best_test_r2 = float("-inf")
    
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


    # This train step function includes gradient accumulation (experimental)
    def train_step_v2(
        self,
        dataloader: torch.utils.data.DataLoader,         
        epoch_number: int = 1,
        amp: bool=True,
        enable_clipping=False,
        accumulation_steps: int = 1,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
    
        """Trains a PyTorch model for a single epoch with gradient accumulation.

        Args:
            dataloader: A DataLoader instance for the model to be trained on.
            epoch_number: Epoch number.
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            accumulation_steps: Number of mini-batches to accumulate gradients before an optimizer step.
                If batch size is 64 and accumulation_steps is 4, gradients are accumulated for 256 mini-batches before an optimizer step.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            train_loss, train_r2.
        """

        # Put model in train mode
        self.model.train()
        #self.model.to(self.device) # Already done in __init__

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = self.scaler

        # Setup train loss and train accuracy values
        len_dataloader = len(dataloader)
        train_loss, train_r2 = 0, 0
        all_y_pred = []
        all_y = []

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
                    #y_pred = y_pred.clamp(1, 100)
                    
                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            self.warning(f"y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")                            
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
                #y_pred = y_pred.clamp(1, 100)
                
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
            y_pred = y_pred.float().view(-1)
            train_r2 += self.calculate_r2(y, y_pred).item()

            all_y_pred.append(y_pred.detach().cpu())
            all_y.append(y.detach().cpu())

            self.clear_cuda_memory(['X', 'y', 'loss'], locals())

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len_dataloader
        train_r2 = train_r2 / len_dataloader
        all_y_pred = torch.cat(all_y_pred).numpy()
        all_y = torch.cat(all_y).numpy()
        
        return train_loss, train_r2, all_y_pred, all_y

    def test_step(
        self,
        dataloader: torch.utils.data.DataLoader,        
        epoch_number: int = 1,
        amp: bool = True,
        debug_mode: bool = False,
        enable_clipping: bool = False
        ) -> Tuple[float, float, float]:
        
        """Tests a PyTorch model for a single epoch.

        Args:
            dataloader: A DataLoader instance for the model to be tested on.            
            epoch_number: Epoch number.
            amp: Whether to use Automatic Mixed Precision for inference.
            debug_mode: Enables logging for debugging purposes.
            enable_clipping: Enables NaN/Inf value clipping for test predictions.

        Returns:
            test_loss, test_r2.
        """

        # Execute the test step is apply_validation is enabled
        if self.apply_validation:

            # Put model in eval mode
            self.model.eval() 
            #self.model.to(self.device) # Already done in __init__

            # Setup test loss and test accuracy values
            len_dataloader = len(dataloader)
            test_loss, test_r2 = 0, 0
            all_y_pred = []
            all_y = []

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():        
                    for batch, (X, y) in enumerate(dataloader):
                        X = X.squeeze(1) if self.squeeze_dim else X
                        y_pred = self.get_predictions(self.model(X.to(self.device)))
                        break
            except RuntimeError:
                inference_context = torch.no_grad()

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
                        #y_pred = y_pred.clamp(1, 100)
                        
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
                    
                    # Calculate and accumulate R2                    
                    y_pred = y_pred.float().view(-1)                       
                    test_r2 += self.calculate_r2(y, y_pred).item()

                    all_y_pred.append(y_pred.detach().cpu())
                    all_y.append(y.detach().cpu())

                    self.clear_cuda_memory(['X', 'y', 'loss'], locals())

            # Adjust metrics to get average loss and accuracy per batch 
            test_loss = test_loss / len_dataloader
            test_r2 = test_r2 / len_dataloader
            all_y_pred = torch.cat(all_y_pred).numpy()
            all_y = torch.cat(all_y).numpy()
        
        # Otherwise set params with initial values
        else:

            test_loss, test_r2, all_y_pred, all_y = self.best_test_loss, self.best_test_r2, None, None
        
        return test_loss, test_r2, all_y_pred, all_y

    # Scheduler step after the optimizer
    def scheduler_step(
        self,
        test_loss: float=None,
        test_r2: float=None
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
                elif self.scheduler.mode == "max" and test_r2 is not None:
                    self.scheduler.step(test_r2)  # Maximize test_r2
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
        test_r2: float = None,        
        epoch: int = None,
    ) -> pd.DataFrame:

        """
        Updates and saves the best model and model_epoch list based on the specified mode(s), as well as the last-epoch model.

        Parameters:
        - test_loss (float, optional): Test loss for the current epoch (used in "loss" mode).        
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
                # R2 criterion
                elif mode == "r2":
                    if test_r2 is None:
                        self.error(f"'test_r2' must be provided when mode is 'r2'.")
                    if test_r2 > self.best_test_r2:
                        remove_previous_best(self.model_name_r2)
                        self.best_test_r2 = test_r2
                        if self.keep_best_models_in_memory:
                            self.model_r2.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_r2)
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
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        scheduler: torch.optim.lr_scheduler=None,        
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
                - "last" (save model at the last epoch),
                - "all" (save models for all epochs),
                - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
            keep_best_models_in_memory (bool, optional): If True, the best models are kept in memory for future inference. The model state from the last epoch will always be kept in memory.
            train_dataloader (torch.utils.data.DataLoader, optional): Dataloader for training the model.
            test_dataloader (torch.utils.data.DataLoader, optional): Dataloader for testing/validating the model.
            apply_validation (bool, optional): Whether to apply validation after each epoch. Default is True.            
            optimizer (torch.optim.Optimizer, optional): Optimizer to use during training.
            loss_fn (torch.nn.Module, optional): Loss function used for training.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler to adjust learning rate during training.            
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
            - test_loss: List of test loss values for each epoch.
            - train_r2: List of training R2 values for each epoch.            
            - test_r2: List of test R2 values for each epoch.            
            - train_time: List of training time for each epoch.
            - test_time: List of testing time for each epoch.
            - lr: List of learning rate values for each epoch.            

        Example output (for 2 epochs):
        {
            epoch: [1, 2],
            train_loss: [2.0616, 1.0537],            
            test_loss: [1.2641, 1.5706],
            train_r2: [0.7626, 0.8100],            
            test_r2: [0.5178, 0.5612],            
            train_time: [1.1234, 1.5678],
            test_time: [0.4567, 0.7890],
            lr: [0.001, 0.0005],            
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
            train_loss, train_r2, train_pred, train_y = self.train_step_v2(
                dataloader=train_dataloader,                
                epoch_number=epoch,
                amp=amp,
                enable_clipping=enable_clipping,
                accumulation_steps=accumulation_steps,
                debug_mode=debug_mode
                )
            train_epoch_time = time.time() - train_epoch_start_time

            # Perform test step and time it
            test_epoch_start_time = time.time()
            test_loss, test_r2, test_pred, test_y = self.test_step(
                dataloader=test_dataloader,                
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
                train_r2=train_r2,
                train_pred=train_pred,
                train_y=train_y,                
                train_epoch_time=train_epoch_time,
                test_loss=test_loss,               
                test_r2=test_r2,                 
                test_pred=test_pred,
                test_y=test_y,
                test_epoch_time=test_epoch_time,
                plot_curves=plot_curves,
            )

            # Scheduler step after the optimizer
            self.scheduler_step(
                test_loss=test_loss,                
                test_r2=test_r2
            )

            # Update and save the best model, model_epoch list based on the specified mode, and the actual-epoch model.
            # If apply_validation is enabled then upate models based on validation results
            df_results = self.update_model(
                test_loss=test_loss if self.apply_validation else train_loss,   
                test_r2=test_r2 if self.apply_validation else train_r2,            
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
        ) -> torch.Tensor:

        """
        Predicts classes for a given dataset using a trained model.

        Args:
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
            dataloader (torch.utils.data.DataLoader): The dataset to predict on.            

        Returns:
            (list): All of the predicted class labels represented by prediction probabilities (softmax)
        """
 
        # Check model to use
        valid_modes =  {"loss", "r2", "last", "all"}
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
        elif model_state == "r2":
            if self.model_r2 is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_r2
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

        y_preds = []
        model.eval()
        model.to(self.device)

        # Set inference context
        try:
            inference_context = torch.inference_mode()
            with torch.inference_mode():
                for batch, (X, y) in enumerate(dataloader):
                    if X.ndimension() == 3 and X.shape[1] == 1:
                        X = X.squeeze(1)
                    check = self.get_predictions(model(X.to(self.device)))
                    break
        except RuntimeError:
            inference_context = torch.no_grad()
            #self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

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
                    if X.ndimension() == 3 and X.shape[1] == 1:  # [batch_size, 1, time_steps]
                        self.squeeze_dim = True
                    elif X.ndimension() == 2:  # [batch_size, time_steps]
                        pass  # No change needed
                    else:
                        self.error(f"Unexpected input shape after exception handling: {X.shape}")
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
                y_pred = self.get_predictions(model(X))               

                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())

        # Concatenate list of predictions into a tensor
        return torch.cat(y_preds)

    def predict_and_store(
        self,
        test_dir: str, 
        transform: Optional[Callable], #Union[torchvision.transforms, torchaudio.transforms],         
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
            sample_fraction (float, optional): The fraction of samples to predict. Defaults to 1.0.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            A list of dictionaries with sample predictions, sample names, prediction probabilities, prediction times, actual labels and prediction classes
            A classification report as a dictionary from sckit-learng.metrics
        """

        # Check model to use
        valid_modes =  {"loss", "r2", "last", "all"}
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
        elif model_state == "r2":
            if self.model_r2 is None:
                self.info(f"Model not found, using last-epoch model for prediction.")
                model = self.model
            else:
                model = self.model_r2                        
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
                    transform = transform.to(self.device)
                    signal = transform(signal)
                except:
                    # Fall back to cpu if error
                    transform = transform.to("cpu")
                    signal = transform(signal)
                
            # Simulate batch
            signal = signal.unsqueeze(0).to(self.device)

            # Prepare model for inference by sending it to target device and turning on eval() mode
            model.to(self.device)
            model.eval()

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():
                    if signal.ndimension() == 3 and signal.shape[1] == 1:
                        check = model(signal.squeeze(1))
                    else:
                        check = model(signal)
            except RuntimeError:
                inference_context = torch.no_grad()
                #self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")
            
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

                # Check the current shape of signal and attempt a fix
                if signal.ndimension() == 3 and signal.shape[1] == 1:  # [batch_size, 1, time_steps]
                    signal = signal.squeeze(1)
                elif signal.ndimension() == 2:  # [batch_size, time_steps]
                    pass  # No change needed
                else:
                    self.error(f"Unexpected input shape after exception handling: {signal.shape}")
            
            # Get prediction probability, predicition label and prediction class
            with inference_context:
                pred = model(signal) # perform inference on target sample             

                # Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
                pred_dict["pred"] = round(pred.unsqueeze(0).max().cpu().item(), 4)                
                
                # End the timer and calculate time per pred
                end_time = timer()
                pred_dict["time_for_pred"] = round(end_time-start_time, 4)

            # Add the dictionary to the list of preds
            pred_list.append(pred_dict)

            clear_output(wait=True)       

        # Generate the classification report
        classification_report_dict = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names="None",
            labels="None",
            output_dict=True
            )

        # Return list of prediction dictionaries
        return pred_list, classification_report_dict
    