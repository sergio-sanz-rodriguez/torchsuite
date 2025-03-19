"""
Contains classes for training and testing a PyTorch model for object detection and segmentation.
"""

import os
import glob
import torch
import time
import numpy as np
import pandas as pd
import copy
import warnings
import re
import torchvision.ops as ops
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Any, List, Union, Optional
from tqdm.auto import tqdm 
from IPython.display import clear_output
from pathlib import Path
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
#from sklearn.metrics import precision_recall_curve, roc_curve, auc
from contextlib import nullcontext
from .common import Common, Colors

import warnings
warnings.filterwarnings("ignore")


# Training and prediction engine class
class ObjectDetectionEngine(Common):

    """
    A class to handle training, evaluation, and predictions for a PyTorch model.

    This class provides functionality to manage the training and evaluation lifecycle
    of a PyTorch model, including saving the best model based on specific criteria.

    Args:
        model (torch.nn.Module, optional): The PyTorch model to handle. Must be instantiated.
        color_map (dict, optional): Specifies the colors for the training and evaluation curves
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
        self.model_loss = None
        self.model_epoch = None
        self.save_best_model = False
        self.keep_best_models_in_memory = False
        self.mode = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.model_name = None
        self.model_name_loss = None        
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
            self.warning(f"Keeping best models in memory may slow down the training process.")
        else:
            self.info(f"Keeping best models in memory: {self.keep_best_models_in_memory}")

    def prepare_data(
        self,
        images,
        targets
        ):

        """
        Prepare the data by performing the following operations:
        1. Moving images to the device (GPU/CPU).
        2. Moving tensor values within the targets dictionary to the device.
        
        Args:
            images (list): List of image tensors.
            targets (list of dict): List of dictionaries, where each dictionary contains target data, 
                                    potentially including tensor values.
        
        Returns:
            tuple: A tuple containing:
                - images (list): List of image tensors moved to the device.
                - targets (list of dict): List of dictionaries, with tensor values moved to the device.
        """
        
        images = list(image.to(self.device) for image in images)            
        targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        return images, targets


    def init_results(self, dictionary: dict = None):

        """Initializes the dictionary for the results"""
        
        dictionary = dictionary or {}  # Ensure dictionary is not None
        self.results.update({
            "epoch": [],
            **{f"train_{key}": [] for key in dictionary.keys()},
            "train_total_loss": [],
            "train_time [s]": [],
            **{f"test_{key}": [] for key in dictionary.keys()},
            "test_total_loss": [],
            "test_time [s]": [],
            "lr": []
        })

    def display_results(
        self,
        epoch: int,
        max_epochs: int,
        train_loss: Dict[str, float],        
        train_epoch_time: float,
        test_loss: Optional[Dict[str, float]] = None,
        test_epoch_time: Optional[float] = None,
        plot_curves: bool = False
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
        
        # Format train loss as a string
        train_loss_str = f"{self.color_other} | ".join(f"{self.color_train}{key}: {value:.4f}" for key, value in train_loss.items())

        print(
            f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
            f"{self.color_train}Train: {self.color_other} {train_loss_str} {self.color_other}| "
            f"{self.color_train}time: {self.sec_to_min_sec(train_epoch_time)} {self.color_other}| "            
            f"{self.color_train}lr: {lr:.10f}"
        )
        if self.apply_validation and test_loss is not None:

            # Format test loss as a string
            test_loss_str = f"{self.color_other} | ".join(f"{self.color_test}{key}: {value:.4f}" for key, value in test_loss.items())

            print(
                f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
                f"{self.color_test}Test:  {self.color_other} {test_loss_str} {self.color_other}| "                
                f"{self.color_test}time: {self.sec_to_min_sec(test_epoch_time)} {self.color_other}| "            
                f"{self.color_test}lr: {lr:.10f}"
            )
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)        
        for key, value in train_loss.items():
            self.results[f"train_{key}"].append(value)
        self.results["train_time [s]"].append(train_epoch_time)
        if test_loss is not None:
            for key, value in test_loss.items():
                self.results[f"test_{key}"].append(value)
            self.results["test_time [s]"].append(test_epoch_time)
        else:
            #`train_loss` always exists, we just need its keys
            for key in train_loss.keys() if train_loss else ["loss"]: 
                self.results[f"test_{key}"].append(None)
            self.results["test_time [s]"].append(None)
        self.results["lr"].append(lr)
        
        # Plot training and test loss curves
        if plot_curves:
            n_plots = len(train_loss.keys())
            cols = min(5, n_plots)
            rows = (n_plots + cols - 1) // cols

            plt.figure(figsize=(25, 6*rows))
            range_epochs = range(1, len(self.results["epoch"]) + 1)

            for i, key in enumerate(train_loss.keys(), start=1):
                plt.subplot(rows, cols, i)
                plt.plot(range_epochs, self.results[f"train_{key}"], label=f"train_{key}", color=self.color_train_plt)
                
                if self.apply_validation and test_loss is not None:
                    plt.plot(range_epochs, self.results[f"test_{key}"], label=f"test_{key}", color=self.color_test_plt)
                
                plt.title(key)
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

        # Validate if the train dataloader has been given
        if not isinstance(dataloader, torch.utils.data.DataLoader) or dataloader is None:
            self.error(f"The train dataloader has incorrect format or is not specified.")

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
        valid_modes = {"loss", "acc", "fpr", "pauc", "last", "all"}
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
            epochs=epochs,
            plot_curves=plot_curves,
            amp=amp,
            enable_clipping=enable_clipping,
            accumulation_steps=accumulation_steps,            
            debug_mode=debug_mode,
            )
        
        # Initialize optimizer, loss_fn, and scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.model.train()

        # Attempt a forward pass to check if the shape of X is compatible
        for batch, (images, targets) in enumerate(dataloader):

            images, targets = self.prepare_data(images, targets)

            try:
                # This is where the model will "complain" if the shape is incorrect
                check = self.model(images, targets)
                # Initialize the log results
                if isinstance(check, dict):
                    self.init_results(check)

            except Exception as e:

                # If the result dict delivered by the model is unknown.
                #if not isinstance(check, dict):
                #    self.error(f"Unknown output metrics from the model.")

                # If the shape is wrong, reshape and try again
                match = re.search(r"got input of size: (\[[^\]]+\])", str(e))
                if match:
                    self.warning(f"Wrong input shape: {match.group(1)}. Attempting to reshape X.")
                else:
                    self.warning(f"Attempting to reshape X.")

                # Check the current shape and attempt a fix
                if images[0].ndimension() == 2:
                    self.squeeze_dim = True
                elif images[0].ndimension() == 3:  # [batch_size, width, height]
                    pass
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
            if "all" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_epoch = []
                    for k in range(epochs):
                        self.model_epoch.append(copy.deepcopy(self.model))
                        self.model_epoch[k].to(self.device)
            self.best_test_loss = float("inf")         
    
    def progress_bar(
        self,
        dataloader: torch.utils.data.DataLoader,
        total: int,
        epoch_number: int,
        stage: str,
        desc_length: int=22):

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
        else:
            color = self.color_test_plt
            desc = f"Validating epoch {epoch_number+1}".ljust(desc_length) + " "
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
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, train_fpr, train_pauc). For example: (0.1112, 0.8743, 0.01123, 0.15561).
        """

        # Put model in train mode
        self.model.train()
        #self.model.to(self.device) # Already done in __init__

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss values
        len_dataloader = len(dataloader)
        train_loss = 0         
        train_loss_dict = {}     

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting
        for batch, (images, targets) in self.progress_bar(dataloader=dataloader, total=len_dataloader, epoch_number=epoch_number, stage='train'):
            
            images, targets = self.prepare_data(images, targets)

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    
                    # Forward pass
                    loss_dict = self.model(images, targets)
                    loss = sum(item for item in loss_dict.values())
                    
                    # Check if the output has NaN or Inf values
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        if enable_clipping:
                            self.warning(f"Loss is NaN or Inf at batch {batch}. Replacing Nans/Infs...")                            
                            loss = torch.nan_to_num(
                                loss,
                                nan=torch.mean(loss).item(), 
                                posinf=torch.max(loss).item(), 
                                neginf=torch.min(loss).item()
                                )
                        else:
                            self.warning(f"Loss is NaN or Inf at batch {batch}. Skipping batch...")
                            continue
                    
                    # Divide into accumulation_steps
                    loss /= accumulation_steps
                                      
                # Backward pass with scaled gradients
                if debug_mode:
                    # Use anomaly detection
                    with torch.autograd.detect_anomaly():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

            else:
                # Forward pass
                loss_dict = self.model(images, targets)
                loss = sum(item for item in loss_dict.values())

                # Divide into accumulation_steps
                loss /= accumulation_steps
                
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
            
            # Accumulate metrics for each loss in loss_dict
            for loss_name, loss_value in loss_dict.items():
                if loss_name in train_loss_dict:
                    train_loss_dict[loss_name] += loss_value.item() * accumulation_steps  # Scale back to original loss
                else:
                    train_loss_dict[loss_name] = loss_value.item() * accumulation_steps
            train_loss += loss.item() * accumulation_steps  # Scale back to original loss

        # Adjust metrics to get average losses per batch
        for loss_name in train_loss_dict:
            train_loss_dict[loss_name] /= len_dataloader                
        train_loss /= len_dataloader

        train_loss_dict.update({"total_loss": train_loss})        

        return train_loss, train_loss_dict

    def test_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch_number: int = 1,
        amp: bool = True,
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
            A tuple of test loss, test accuracy, FPR-at-recall, and pAUC-at-recall metrics.
        """

        # Execute the test step is apply_validation is enabled
        if self.apply_validation:

            # Put model in train mode, otherwise loss results are not generated
            self.model.train() 

            # Setup test loss and test accuracy values
            len_dataloader = len(dataloader)
            test_loss = 0
            test_loss_dict = {} 

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():        
                    for batch, (images, targets) in enumerate(dataloader):

                        images, targets = self.prepare_data(images, targets)
                        check = self.model(images, targets)
                        break
            except RuntimeError:
                inference_context = torch.no_grad()
                self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

            # Turn on inference context manager 
            with inference_context:

                # Loop through DataLoader batches                
                for batch, (images, targets) in self.progress_bar(dataloader=dataloader, total=len_dataloader, epoch_number=epoch_number, stage='test'):

                    images, targets = self.prepare_data(images, targets)

                    # Enable AMP if specified
                    with torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext():

                         # Forward pass
                        loss_dict = self.model(images, targets)
                        loss = sum(item for item in loss_dict.values())

                        # Check if the output has NaN or Inf values
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            if enable_clipping:
                                self.warning(f"Loss is NaN or Inf at batch {batch}. Replacing Nans/Infs...")                            
                                loss = torch.nan_to_num(
                                    loss,
                                    nan=torch.mean(loss).item(), 
                                    posinf=torch.max(loss).item(), 
                                    neginf=torch.min(loss).item()
                                    )
                            else:
                                self.warning(f"Loss is NaN or Inf at batch {batch}. Skipping batch...")
                                continue

                         # Accumulate individual component losses
                        for loss_name, loss_value in loss_dict.items():
                            if loss_name in test_loss_dict:
                                test_loss_dict[loss_name] += loss_value.item()
                            else:
                                test_loss_dict[loss_name] = loss_value.item()
                        test_loss += loss.item()                    

            # Adjust metrics to get average losses per batch 
            for loss_name in test_loss_dict:
                test_loss_dict[loss_name] /= len_dataloader
            test_loss /= len_dataloader

            test_loss_dict.update({"total_loss": test_loss})

        # Otherwise set params with initial values
        else:            
            test_loss = None
            test_loss_dict = None            

        return test_loss, test_loss_dict


    # Scheduler step after the optimizer
    def scheduler_step(
        self,
        test_loss: float=None,
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
        - Saves the best-performing model during training based on the specified testuation mode.
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
        Finalizes the training process by closing writer and showing the elapsed time.
        
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
            test_dataloader (torch.utils.data.DataLoader, optional): Dataloader for testing or validating the model.
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
            - train_<loss_metrics>: List of training loss values for each epoch.            
            - test_<loss_metrics>: List of test loss values for each epoch.
            - train_time: Training_time for each epcoh            
            - test_time: Testing time for each epoch.
            - lr: Learning rate value for each epoch.

        Example output (for 2 epochs):
        {
            epoch: [1, 2],
            train_loss: [2.0616, 1.0537],            
            test_loss: [1.2641, 1.5706],            
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
            train_loss, train_loss_dict = self.train_step_v2(
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
            test_loss, test_loss_dict = self.test_step(
                dataloader=test_dataloader,            
                epoch_number=epoch,
                amp=amp,
                enable_clipping=enable_clipping,
                )
            test_epoch_time = time.time() - test_epoch_start_time if self.apply_validation else 0.0            

            clear_output(wait=True)

            # Show results
            self.display_results(
                epoch=epoch,
                max_epochs=epochs,
                train_loss=train_loss_dict,                
                train_epoch_time=train_epoch_time,
                test_loss=test_loss_dict,                
                test_epoch_time=test_epoch_time,
                plot_curves=plot_curves,
            )

            # Scheduler step after the optimizer
            self.scheduler_step(
                test_loss=test_loss,
            )

            # Update and save the best model, model_epoch list based on the specified mode, and the actual-epoch model.
            # If apply_validation is enabled then upate models based on validation results
            df_results = self.update_model(
                test_loss=test_loss if self.apply_validation or test_loss is not None else train_loss,                
                epoch=epoch
                )

        # Finish training process
        train_time = time.time() - train_start_time
        self.finish_train(train_time)

        return df_results

    @staticmethod
    def prune_predictions(
        pred,
        score_threshold=0.66,
        mask_threshold=0.5,
        iou_threshold=0.5,
        ):

        """
        Filters and refines predictions by:
        1. Removing low-confidence detections based on the score threshold.
        2. Applying a binary mask threshold to filter out weak segmentation masks.
        3. Using Non-Maximum Suppression (NMS) to eliminate overlapping predictions.

        Parameters:
        pred : dict
            The raw predictions containing "boxes", "scores", "labels", and "masks".
        score_threshold : float, optional
            The minimum confidence score required to keep a prediction (default: 0.7).
        mask_threshold : float, optional
            The threshold for binarizing the segmentation masks (default: 0.5).
        iou_threshold : float, optional
            The Intersection over Union (IoU) threshold for NMS (default: 0.5).

        Returns:
        dict
            A dictionary with filtered and refined predictions:
            - "boxes": Tensor of kept bounding boxes.
            - "scores": Tensor of kept scores.
            - "labels": List of label strings.
            - "masks": Tensor of kept segmentation masks. [OPTIONAL]
        """
        
        # Filter predictions based on confidence score threshold
        scores = pred["scores"]
        high_conf_idx = scores > score_threshold

        filtered_pred = {
            "boxes":  pred["boxes"][high_conf_idx].long(),
            "scores": pred["scores"][high_conf_idx],
            "labels": pred["labels"][high_conf_idx], #[f"roi: {s:.3f}" for s in scores[high_conf_idx]]
        }

        # Only add "masks" if present in prediction output
        if "masks" in pred:
            filtered_pred["masks"] = (pred["masks"] > mask_threshold).squeeze(1)[high_conf_idx]

        # Apply Non-Maximum Suppression (NMS) to remove overlapping predictions
        if len(filtered_pred["boxes"]) == 0:
            return filtered_pred  # No boxes to process
        keep_idx = ops.nms(filtered_pred["boxes"].float(), filtered_pred["scores"], iou_threshold)

        # Return filtered predictions
        return {
            "boxes": filtered_pred["boxes"][keep_idx],
            "scores": filtered_pred["scores"][keep_idx],
            "labels": filtered_pred["labels"][keep_idx], #[i] for i in keep_idx],
            **({"masks": filtered_pred["masks"][keep_idx]} if "masks" in filtered_pred else {})  # Only add "masks" if present
        }

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_state: str="last",
        prune_predictions: bool=True,
        **kwargs
        ):

        """
        Predicts classes for a given dataset using a trained model.

        Args:
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
            dataloader (torch.utils.data.DataLoader): The dataset to predict on.            
            prune_predictions (bool): A flag to activate a method that filters out redundant ROIs.
            **kwargs: Additional arguments to pass to the apply_postprocessing function, such as thresholds.         

        Returns:
            (list): All of the predicted class labels represented by prediction probabilities (softmax)
        """
 
        # Check model to use
        valid_modes =  {"loss", "last", "all"}
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
                for images, targets in dataloader:
                    
                    images, targets = self.prepare_data(images, targets)
                    y_pred = model(images)
                    break
        except RuntimeError:
            inference_context = torch.no_grad()
            self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

        # Turn on inference context manager 
        with inference_context:
            for images, targets in tqdm(dataloader, desc="Making predictions"):

                # Send data and targets to target device
                images, targets = self.prepare_data(images, targets)
                                
                # Do the forward pass
                y_pred = model(images)

                # If prune_predictions is enabled, apply it to filter predictions
                # Process predictions: prune if enabled, format labels
                y_pred = [
                    self.prune_predictions(pred, **kwargs) if prune_predictions else pred
                    for pred in y_pred
                ]

                y_preds.extend(y_pred)

        # Convert predictions to CPU
        cpu_preds = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in pred.items()} for pred in y_preds]
        
        # Concatenate list of predictions into a tensor
        return cpu_preds