"""
Contains classes for training and testing a PyTorch model for image segmentation.
"""

import os
import glob
import torch
import time
import numpy as np
import pandas as pd
import copy
import warnings
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
from contextlib import nullcontext
from .common import Common, Colors

import warnings
warnings.filterwarnings("ignore")


# Training and prediction engine class
class SegmentationEngine(Common):

    """
    A class to handle training, evaluation, and predictions for a PyTorch model.

    This class provides functionality to manage the training and evaluation lifecycle
    of a PyTorch model, including saving the best model based on specific criteria.

    Args:
        model (torch.nn.Module, optional): The PyTorch model to handle. Must be instantiated.
        color_map (dict, optional): Specifies the colors for the training and evaluation curves
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
        self.model_loss = None
        self.model_dice = None
        self.model_iou = None
        self.model_epoch = None
        self.save_best_model = False
        self.keep_best_models_in_memory = False
        self.mode = None
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.model_name = None
        self.model_name_loss = None        
        self.model_name_dice = None
        self.model_name_iou = None
        self.num_classes = 1
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
        assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
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

        # Create the model path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
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
            "train_loss": [],
            "train_dice": [],
            "train_iou": [],
            "train_time [s]": [],            
            "test_loss": [],
            "test_dice": [],
            "test_iou": [],
            "test_time [s]": [],
            "lr": []
        })

    def display_results(
        self,
        epoch: int,
        max_epochs: int,
        train_loss: Dict[str, float],
        train_dice: Dict[str, float],        
        train_iou: Dict[str, float],
        train_epoch_time: float,
        test_loss: Optional[Dict[str, float]] = None,
        test_dice: Optional[Dict[str, float]] = None,
        test_iou: Optional[Dict[str, float]] = None,
        test_epoch_time: Optional[float] = None,
        plot_curves: bool = False
        ):
    
        """
        Displays the training and validation results both numerically and visually.

        Functionality:
        - Outputs key metrics such as training and validation loss, accuracy, and fpr at recall in numerical form.
        - Generates plots that visualize the training process, such as:
        - Loss curves (training vs validation loss over epochs).
        - Dice-coefficient curves (training vs validation accuracy over epochs).
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
            f"{self.color_train}dice: {train_dice:.4f} {self.color_other}| "
            f"{self.color_train}iou: {train_iou:.4f} {self.color_other}| "
            f"{self.color_train}time: {self.sec_to_min_sec(train_epoch_time)} {self.color_other}| "            
            f"{self.color_train}lr: {lr:.10f}"
        )
        if self.apply_validation:
            print(
                f"{self.color_other}Epoch: {epoch+1}/{max_epochs} | "
                f"{self.color_test}Test:  {self.color_other}| "
                f"{self.color_test}loss: {test_loss:.4f} {self.color_other}| "
                f"{self.color_test}dice: {test_dice:.4f} {self.color_other}| "
                f"{self.color_test}iou: {test_iou:.4f} {self.color_other}| "
                f"{self.color_test}time: {self.sec_to_min_sec(test_epoch_time)} {self.color_other}| "            
                f"{self.color_test}lr: {lr:.10f}"
            )
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)
        self.results["train_loss"].append(train_loss)
        self.results["train_dice"].append(train_dice)
        self.results["train_iou"].append(train_iou)
        self.results["test_loss"].append(test_loss)
        self.results["test_dice"].append(test_dice)
        self.results["test_iou"].append(test_iou)
        self.results["train_time [s]"].append(train_epoch_time)
        self.results["test_time [s]"].append(test_epoch_time)
        self.results["lr"].append(lr)
        
        # Plots training and test loss, accuracy, and fpr-at-recall curves.
        if plot_curves:
        
            n_plots = 3
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

            # Plot dice
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, self.results["train_dice"], label="train_dice_coeff", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_dice"], label="test_dice_coeff", color=self.color_test_plt)
            plt.title("Dice-Coefficient")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot iou
            plt.subplot(1, n_plots, 3)
            plt.plot(range_epochs, self.results["train_iou"], label="train_iou", color=self.color_train_plt)
            if self.apply_validation:
                plt.plot(range_epochs, self.results["test_iou"], label="test_iou", color=self.color_test_plt)
            plt.title("Intersection over Union (IoU)")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()            
                    
            plt.show()

    def init_train(
        self,
        target_dir: str=None,
        model_name: str=None,
        dataloader: torch.utils.data.DataLoader=None,
        num_classes: int=1,
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
                - "dice": saves the epoch with the highest validation dice coefficient
                - "iou": saves the epoch with the highest validation iou
                - "all": saves models for all epochs
                - A list, e.g., ["loss", "dice"], is also allowed. Only applicable if `save_best_model` is True.
                - None: the model will not be saved.            

        Functionality:
            Validates `accumulation_steps` and `epochs` and other parameters with assertions.
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
        valid_modes = {"loss", "dice", "iou", "last", "all"}
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
                classes_in_mask = check.shape[1]  # dim=1 is the num_class dimension for the mask                              
                if isinstance(check, torch.Tensor) and num_classes == classes_in_mask:
                    self.num_classes = num_classes
            except Exception as e:
                raise ValueError(r"Unexpected error when checking the model", str(e))
            break
    
        # Initialize the best model and model_epoch list based on the specified mode.
        if self.save_best_model:
            if "loss" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_loss = copy.deepcopy(self.model)                            
                    self.model_loss.to(self.device)
                self.model_name_loss = self.model_name.replace(".", f"_loss.")
            if "dice" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_dice = copy.deepcopy(self.model)                            
                    self.model_dice.to(self.device)
                self.model_name_dice = self.model_name.replace(".", f"_dice.")
            if "iou" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_iou = copy.deepcopy(self.model)
                    self.model_iou.to(self.device)
                self.model_name_iou = self.model_name.replace(".", f"_iou.")            
            if "all" in self.mode:
                if self.keep_best_models_in_memory:
                    self.model_epoch = []
                    for k in range(epochs):
                        self.model_epoch.append(copy.deepcopy(self.model))
                        self.model_epoch[k].to(self.device)
            self.best_test_loss = float("inf") 
            self.best_test_dice = 0.0
            self.best_test_iou = 0.0               
    
    
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
            A tuple of training loss, training dice, training_iou.
            In the form (train_loss, train_dice, train_iou).
        """

        self.info(f"Training epoch {epoch_number+1}...")

        # Put model in train mode
        self.model.train()
        #self.model.to(self.device) # Already done in __init__

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup metric values        
        len_dataloader = len(dataloader)
        train_loss, train_dice, train_iou = 0, 0, 0

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len_dataloader, colour=self.color_train_plt):
            
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
                    loss = self.loss_fn(y, y_pred) / accumulation_steps
                
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
                loss = self.loss_fn(y, y_pred) / accumulation_steps

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
            train_dice += self.dice_coefficient(y, y_pred, self.num_classes) # This returns a cpu scalar
            train_iou += self.intersection_over_union(y, y_pred, self.num_classes) # This returns a cpu scalar

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len_dataloader
        train_dice = train_dice / len_dataloader
        train_iou = train_iou / len_dataloader

        return train_loss, train_dice, train_iou

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
            A tuple of test loss, test accuracy, FPR-at-recall, and pAUC-at-recall metrics.
        """

        # Execute the test step is apply_validation is enabled
        if self.apply_validation:

            self.info(f"Validating epoch {epoch_number+1}...")

            # Put model in eval mode
            self.model.eval() 
            #self.model.to(self.device) # Already done in __init__

            # Setup test metric values
            len_dataloader = len(dataloader)
            test_loss, test_dice, test_iou = 0, 0, 0
            
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
                for batch, (X, y) in tqdm(enumerate(dataloader), total=len_dataloader, colour=self.color_test_plt):
                    
                    # Send data to target device
                    X, y = X.to(self.device), y.to(self.device)                    
                                        
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
                        loss = self.loss_fn(y, y_pred)
                        test_loss += loss.item()

                        # Debug NaN/Inf loss
                        if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                            self.warning(f"Loss is NaN/Inf at batch {batch}. Skipping...")
                            continue

                    # Calculate and accumulate accuracy
                    y_pred = y_pred.float() # Convert to float for stability
                    test_dice += self.dice_coefficient(y, y_pred, self.num_classes) # This returns a cpu scalar
                    test_iou += self.intersection_over_union(y, y_pred, self.num_classes) # This returns a cpu scalar

            # Adjust metrics to get average loss and accuracy per batch 
            test_loss = test_loss / len_dataloader
            test_dice = test_dice / len_dataloader
            test_iou = test_iou / len_dataloader
        
        # Otherwise set params with initial values
        else:
            test_loss, test_dice, test_iou = self.best_test_loss, self.best_test_dice, self.best_test_iou

        return test_loss, test_dice, test_iou


    # Scheduler step after the optimizer
    def scheduler_step(
        self,
        test_loss: float=None,
        test_dice: float=None,
        ):

        """
        Performs a scheduler step after the optimizer step.

        Parameters:
        - scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        - test_loss (float, optional): Test loss value, required for ReduceLROnPlateau with 'min' mode.
        - test_dice (float, optional): Test dice-coefficient value, required for ReduceLROnPlateau with 'max' mode.
        """
            
        if self.scheduler:
            if self.apply_validation and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Check whether scheduler is configured for "min" or "max"
                if self.scheduler.mode == "min" and test_loss is not None:
                    self.scheduler.step(test_loss)  # Minimize test_loss
                elif self.scheduler.mode == "max" and test_dice is not None:
                    self.scheduler.step(test_dice)  # Maximize test_accuracy
                else:
                    self.error(
                        f"The scheduler requires either `test_loss` or `test_dice` "
                        "depending on its mode ('min' or 'max')."
                        )
            else:
                self.scheduler.step()  # For other schedulers
    
    # Updates and saves the best model and model_epoch list based on the specified mode.
    def update_model(
        self,
        test_loss: float = None,
        test_dice: float = None,
        test_iou: float = None,        
        epoch: int = None,
    ) -> pd.DataFrame:

        """
        Updates and saves the best model and model_epoch list based on the specified mode(s), as well as the last-epoch model.

        Parameters:
        - test_loss (float, optional): Test loss for the current epoch (used in "loss" mode).
        - test_dice (float, optional): Test dice coefficient for the current epoch (used in "dice" mode).
        - test_iou (float, optional): Test IoU for the current epoch (used in "iou" mode).        
        - epoch (int, optional): Current epoch index, used for naming models when saving all epochs in "all" mode.

        Functionality:
        - Saves the last-epoch model.
        - Saves the logs (self.results).
        - Saves the best-performing model during training based on the specified evaluation mode.
        - If mode is "all", saves the model for every epoch.
        - Updates `self.model_<loss, dice, iou, epoch>` accordingly.

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
                # Dice-coefficient criterion    
                elif mode == "dice":
                    if test_dice is None:
                        self.error(f"'test_dice' must be provided when mode is 'dice'.")
                    if test_dice > self.best_test_dice:
                        remove_previous_best(self.model_name_dice)
                        self.best_test_dice = test_dice
                        if self.keep_best_models_in_memory:
                            self.model_dice.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_dice)
                # IoU criterion    
                elif mode == "iou":
                    if test_iou is None:
                        self.error(f"'test_iou' must be provided when mode is 'iou'.")
                    if test_iou > self.best_test_iou:
                        remove_previous_best(self.model_name_iou)
                        self.best_test_iou = test_iou
                        if self.keep_best_models_in_memory:
                            self.model_iou.load_state_dict(self.model.state_dict())
                        save_model(self.model_name_iou)
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
        num_classes: int=2, 
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
                - "dice" (validation Dice-coefficient),
                - "iou" (validation IoU),                
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
            - train_dice: List of training dice-coefficient values for each epoch.
            - train_iou: List of training IoU values for each epoch.
            - test_loss: List of test loss values for each epoch.
            - test_dice: List of test dice-coefficient values for each epoch.
            - test_iou: List of test IoU values for each epoch.
            - train_time: List of training time for each epoch.
            - test_time: List of testing time for each epoch.
            - lr: List of learning rate values for each epoch.

        Example output (for 2 epochs):
        {
            epoch: [1, 2],
            train_loss: [2.0616, 1.0537],
            train_dice: [0.3945, 0.3945],
            train_iou: [0.4415, 0.5015],
            test_loss: [1.2641, 1.5706],
            test_dice: [0.3400, 0.2973],
            test_iou: [0.4174, 0.3481],
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
            num_classes=num_classes,
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
            train_loss, train_dice, train_iou = self.train_step_v2(
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
            test_loss, test_dice, test_iou = self.test_step(
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
                train_dice=train_dice,
                train_iou=train_iou,                                
                train_epoch_time=train_epoch_time,
                test_loss=test_loss,
                test_dice=test_dice,
                test_iou=test_iou,                
                test_epoch_time=test_epoch_time,
                plot_curves=plot_curves,                
            )

            # Scheduler step after the optimizer
            self.scheduler_step(
                test_loss=test_loss,
                test_dice=test_dice
            )

            # Update and save the best model, model_epoch list based on the specified mode, and the actual-epoch model.
            # If apply_validation is enabled then upate models based on validation results
            df_results = self.update_model(
                test_loss=test_loss if self.apply_validation else train_loss,
                test_dice=test_dice if self.apply_validation else train_dice,
                test_iou=test_iou if self.apply_validation else train_iou,                
                epoch=epoch
                )

        # Finish training process
        train_time = time.time() - train_start_time
        self.finish_train(train_time)

        return df_results