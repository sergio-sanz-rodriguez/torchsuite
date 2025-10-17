"""
Contains classes for training and testing a PyTorch model for classification tasks.  
"""

import os
import glob
import torch
import random
import time
import pandas as pd
import copy
import warnings
import gzip
import inspect
import matplotlib.pyplot as plt
import torchvision.ops as ops
import torch.nn.functional as F
from . import loss_functions
from matplotlib.ticker import FuncFormatter
from .common import Common, Colors
from tqdm.auto import tqdm 
from PIL import Image
from pathlib import Path
from typing import List, Union
from IPython.display import clear_output, display, HTML
from timeit import default_timer as timer
from contextlib import nullcontext
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

            
# Training and prediction engine class
class SegmentationEngine(Common):

    def __init__(
        self,
        model: torch.nn.Module=None,        
        optimizer: torch.optim.Optimizer=None,
        loss_fn: torch.nn.Module=None,
        scheduler: torch.optim.lr_scheduler=None,
        color_map: dict=None,
        theme: str='light',
        log_verbose: bool=True,
        device: str="cuda" if torch.cuda.is_available() else "cpu"
        ):

        """
        A class to handle training, evaluation, and predictions for a PyTorch model.

        This class provides functionality to manage the training and evaluation lifecycle
        of a PyTorch model, including saving the best model based on specific criteria.

        Args:
            model (torch.nn.Module): PyTorch model to train. It must be instantiated.        
            optimizer (torch.optim.Optimizer, for .train() only): The optimizer to minimize the loss function.
            loss_fn (torch.nn.Module, for .train() only): The loss function to minimize during training.            
            scheduler (torch.optim.lr_scheduler, optional, for .train() only): Learning rate scheduler for the optimizer. If None, LR is fixed.
            color_map (dict, optional): Specifies the colors for the training and evaluation curves:
                'black', 'off-black' or 'very_dark_gray', 'blue', 'orange', 'green', 'red', 'yellow', 
                'magenta', 'cyan', 'white', 'light_gray', 'dark_gray', 'light_blue', 'light_green',
                'light_red', 'light_yellow', 'light_magenta', 'light_cyan'.
                Example: {'train': 'blue', 'test': 'orange', 'other': 'black'}
                Note: if some keys are not specified, their default colors will be used based on the theme argument.
            theme (str, optional): Theme for logging and plot visualization.
                Must be either 'light' or 'dark'. Affects background, text, and grid colors.
            log_verbose (bool, optional): if True, activate logger messages.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        
        Public API:
            .load(...): Loads a PyTorch model into the engine from a target directory and optionally returns it. Engine must be instantiated beforehand.
            .save(...): Saves a PyTorch model to a target directory.
            .train(...): Trains and tests a PyTorch model for a given number of epochs.
            .predict(...): Predicts classes for a given dataset using a trained model.
        """

        super().__init__(theme=theme, log_verbose=log_verbose)

        # Initialize self variables
        self.model = model
        self.optimizer = optimizer                
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.log_verbose = log_verbose
        self.device = device
        self._set_colormap(color_map, theme)
        self.model_loss = None
        self.model_dice = None
        self.model_iou = None
        self.model_epoch = None
        self.save_best_model = False
        self.keep_best_models_in_memory = False
        self.mode = None
        self.model_name = None
        self.model_name_loss = None        
        self.model_name_dice = None
        self.model_name_iou = None  
        self.checkpoint_path_prefix = "ckpt"
        self.valid_modes =  {"loss", "dice", "iou", "last", "all"}
        
        # Check if model is provided
        if self.model is None:
            self.error(f"Instantiate the engine by passing a PyTorch model to handle.")
        else:
            self.model.to(self.device)  


    # ======================================= #
    # ========== UTILITY / LOGGING ========== #
    # ======================================= #

    # Function to initialize the result logs
    def _init_results(
            self,
            dictionary: dict = None
            ):

        """
        Initializes the dictionary for the results-

        Args:
            dictionary: A dictionary with train and test lossees, lr, and times.
        """
        self.results = {}
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

        if self.plot_curves:
            self.metric_labels = {
                'loss':      ['loss', 'loss', 'Loss'],             
                'dice':      ['dice', 'dice_coeff', 'Dice-Coefficient'],
                'iou':       ['iou', 'iou', 'Intersection over Union (IoU)'],
                'time':      ['time [s]', 'time', 'Time [MMmSSs]'],                
                'lr':        ['lr', 'lr', 'Learning Rate']    
            }

    # Function to set the settings and line width for visualization
    def _set_colormap(
        self,
        color_map,
        theme
        ):

        """
        Initializes the color settings and line width for visualization.

        Args:
            color_map (dict or None): A dictionary specifying custom colors for 'train', 'test', and 'other' categories.
                If None, default colors will be used based on the theme.
            theme : (str): Visualization theme to apply. Must be either 'light' or 'dark'.
                Affects background, text, and grid colors.
        """

        # Validate and set the theme (default to 'light' if invalid)
        if isinstance(theme, str) and theme.lower() in ['light', 'dark']:
            self.theme = theme.lower()
        else:
            self.theme = 'light'

        # Set default color mappings based on the theme
        if self.theme == 'light':
            default_color_map = {'train': 'blue', 'test': 'orange', 'other': 'black'}
            self.figure_color_map = {'bg': 'white', 'text': 'black', 'grid': '#b0b0b0'} #cccccc
        else:
            default_color_map = {'train': 'yellow', 'test': 'light_red', 'other': 'white'}
            self.figure_color_map = {'bg': '#1e1e1e', 'text': 'white', 'grid': '#666666'}

        # Merge user-defined color_map with defaults (user values override defaults)
        if color_map is None:
            color_map = default_color_map # Use defaults if no user input
        else:
            user_map = color_map.copy()
            # Set 'other' if user hasn't defined it
            if 'other' not in user_map:
                user_map['other'] = default_color_map['other']
            color_map = {**default_color_map, **user_map} # Merge user input with defaults
            
        # Convert color names to console-friendly and matplotlib-compatible formats
        self.color_train = Colors.get_console_color(color_map['train'])
        self.color_test =  Colors.get_console_color(color_map['test'])
        self.color_other = Colors.get_console_color(color_map['other'])        
        self.color_train_plt = Colors.get_matplotlib_color(color_map['train'])
        self.color_test_plt =  Colors.get_matplotlib_color(color_map['test'])     
        #self.color_other_plt = Colors.get_matplotlib_color(color_map['other'] if theme=='light' else 'dark_gray')
                    
        # Set the default line width for plots
        self.linewidth = Colors.get_linewidth()

    # Funtion to display the training configuration parameters
    def _print_config(
        self,
        batch_size,
        resume_msg
        ):
        
        """
        Prints the configuration of the training process.

        Args:
            batch_size (int): Batch size
            resume_msg (str, optional): An additional string with information on resume
        """

        if self.resume and self.start_epoch > 0:
            self.info(f"Overriding arguments...")
            if isinstance(resume_msg, (str)) and len(resume_msg) > 0:
                self.info(f"Resume: {resume_msg}")
            else:
                self.info(f"Resume: {self.resume}")
        else:
            self.info(f"Resume: {self.resume}")                
        self.info(f"Device: {self.device}")
        self.info(f"Epochs: {self.epochs}")
        self.info(f"Batch size: {batch_size}")
        self.info(f"Accumulation steps: {self.accumulation_steps}")
        self.info(f"Effective batch size: {batch_size * self.accumulation_steps}")
        self.info(f"Augmentation strategy: {self.augmentation_strategy}")
        if self.augmentation_strategy == "off_last":
            self.info(f"Augmentation off epochs: last {self.augmentation_off_epochs}")
        if self.augmentation_strategy == "random":
            self.info(f"Augmentation probability: {self.augmentation_random_prob}")
        self.info(f"Initial learning rate: {self.optimizer.param_groups[0]['lr']}")        
        self.info(f"Apply validation: {self.apply_validation}")        
        self.info(f"Plot curves: {self.plot_curves}")
        self.info(f"Automatic Mixed Precision (AMP): {self.amp}")
        self.info(f"Enable clipping: {self.enable_clipping}")
        self.info(f"Debug mode: {self.debug_mode}")        
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
    
    # Visualization helper function to print out the results
    def _format_epoch_results(
        self,
        split: str
        ):

        """
        Format training/validation/test metrics into a colored log string.

        Args:
            split (str): Dataset split name (e.g., "Train", "Test").

        Returns:
            str: Formatted string with metrics for the given split.

        """

        split = split.lower()
        if split == 'train':
            return (
                f"{self.color_train}Epoch: {self.results['epoch'][-1]}/{self.epochs} {self.color_other}| "
                f"{self.color_train}Train {self.color_other}| "
                f"{self.color_train}loss: {self.results[f'{split}_loss'][-1]:.4f} {self.color_other}| "                
                f"{self.color_train}dice: {self.results[f'{split}_dice'][-1]:.4f} {self.color_other}| "
                f"{self.color_train}iou: {self.results[f'{split}_iou'][-1]:.4f} {self.color_other}| "
                f"{self.color_train}time: {self.sec_to_min_sec(self.results[f'{split}_time [s]'][-1])} {self.color_other}| "            
                f"{self.color_train}lr: {self.results['lr'][-1]:.10f}"
            )
        else:
            return (
                f"{self.color_test}Epoch: {self.results['epoch'][-1]}/{self.epochs} {self.color_other}| "
                f"{self.color_test}Test  {self.color_other}| "
                f"{self.color_test}loss: {self.results[f'{split}_loss'][-1]:.4f} {self.color_other}| "                
                f"{self.color_test}dice: {self.results[f'{split}_dice'][-1]:.4f} {self.color_other}| "
                f"{self.color_test}iou: {self.results[f'{split}_iou'][-1]:.4f} {self.color_other}| "
                f"{self.color_test}time: {self.sec_to_min_sec(self.results[f'{split}_time [s]'][-1])} {self.color_other}| "            
                f"{self.color_test}lr: {self.results['lr'][-1]:.10f}"
            )

    # Visualization helper function to plot the results
    def _plot(
        self,
        ax,
        range_epochs,
        metric
        ):

        """
        Plots the evolution of a specified training or validation metric over epochs.

        This method handles both standard numerical metrics (like loss or accuracy)
        and time-based metrics (in seconds), formatting the y-axis accordingly.

        Args:
            ax (matplotlib.axes.Axes): The subplot axis to draw on.
            range_epochs (iterable): The range of epochs (x-axis).
            metric (str): The name of the metric to plot. If 'lr', plots the learning rate.
                        If 'time' is in the name, formats y-axis using MM:SS style.
        """

        if metric in ['loss', 'dice', 'iou', 'time']:
            idx_train = f"train_{self.metric_labels[metric][0]}"
            idx_test = f"test_{self.metric_labels[metric][0]}"
            label_train = f"train_{self.metric_labels[metric][1]}"
            label_test = f"test_{self.metric_labels[metric][1]}"
            title = self.metric_labels[metric][2]
            marker = 'o' if range_epochs[-1] == 1 else ''
            ax.plot(range_epochs, self.results[idx_train], label=label_train, color=self.color_train_plt, linewidth=self.linewidth, marker=marker)
            if self.apply_validation:
                ax.plot(range_epochs, self.results[idx_test], label=label_test, color=self.color_test_plt, linewidth=self.linewidth, marker=marker)
            ax.set_title(title, color=self.figure_color_map['text'])
            if metric == 'time':
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: self.sec_to_min_sec(x)))            
            ax.set_xlabel("Epochs", color=self.figure_color_map['text'])        
        else: #lr
            idx = self.metric_labels[metric][0]
            label = self.metric_labels[metric][1]
            title = self.metric_labels[metric][2]
            marker = 'o' if range_epochs[-1] == 1 else ''
            ax.plot(range_epochs, self.results[idx], label=label, color=self.color_train_plt, linewidth=self.linewidth, marker=marker)
            ax.set_title(title, color=self.figure_color_map['text'])
            ax.set_xlabel("Epochs", color=self.figure_color_map['text'])

        ax.tick_params(axis='x', colors=self.figure_color_map['text'])
        ax.tick_params(axis='y', colors=self.figure_color_map['text'])            
        ax.grid(visible=True, which="both", axis="both", color=self.figure_color_map['grid'], alpha=0.8)
        ax.set_facecolor(self.figure_color_map['bg']) 

        legend = ax.legend()
        legend.get_frame().set_facecolor(self.figure_color_map['bg'])
        legend.get_frame().set_edgecolor(self.figure_color_map['text'])

        for spine in ax.spines.values():
            spine.set_color(self.figure_color_map['text'])
        for text in legend.get_texts():
            text.set_color(self.figure_color_map['text'])

    # Function to validate the format of the dataloaders
    def _validate_dataloaders(self):

        """
        Validates the format of the dataloaders as a dictionary with three keys: 'train', 'test', 'train_aug_off'.
        """

        # Validate dataloaders
        if self.dataloaders is None:
            self.error(f"'dataloaders' must be provided as a dictionary with fields: 'train', 'train_aug_off' (opt), 'test'.")
        
        # 'train' is always required
        if "train" not in self.dataloaders or not isinstance(self.dataloaders["train"], torch.utils.data.DataLoader):
            self.error(f"'dataloaders' must contain key 'train' with a DataLoader instance.")
        
        # 'test' is required only if apply_validation is True        
        if self.apply_validation:
            if "test" not in self.dataloaders or not isinstance(self.dataloaders["test"], torch.utils.data.DataLoader):
                self.error(f"'dataloaders' must contain key 'test' with a DataLoader instance when apply_validation=True.")
        
        # 'train_aug_off' is required for the off_epochs and random augmentation strategies
        if self.augmentation_strategy != "always":
            if "train_aug_off" not in self.dataloaders or not isinstance(self.dataloaders["train_aug_off"], torch.utils.data.DataLoader):
                self.error(f"'dataloaders' must contain key 'train_aug_off' for the selected augmentation strategy: {self.augmentation_strategy}.")

    # Utility function for _progress_bar
    def _inject_notebook_tqdm_style(self):

        """
        Inject CSS fix for notebook tqdm bar background in dark themes.
        """

        style = """
        <style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }  
        </style>
        """
        display(HTML(style))

    # Utility function for _progress_bar
    def _is_notebook(self):

        """
        Check if running in a Jupyter notebook.
        """

        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell == 'ZMQInteractiveShell'
        except:
            return False
        
    # Display progress bar
    def _progress_bar(
        self,
        dataloader: torch.utils.data.DataLoader,
        total: int,
        stage: str,
        epoch: int = 1,
        desc_length: int = 22):

        """
        Creates the tqdm progress bar for the training and validation stages.

        Args:
            dataloader: The dataloader for the current stage.
            total: Length of the dataloader
            stage: The current stage ("train" or "validate").
            epoch: The current epoch number.            
            desc_length: The length of the description string for the progress bar.

        Returns:
            A tqdm progress bar instance for the current stage.
        """

        # Inject CSS to fix tqdm notebook theme-based background (only once per call)
        if self._is_notebook():
            self._inject_notebook_tqdm_style()
        
        # Set messages and color
        train_str = f"Training epoch {epoch+1}"
        val_str = f"Validating epoch {epoch+1}"
        if stage == 'train':
            color = self.color_train_plt
            desc = f"Training epoch {epoch+1}".ljust(desc_length) + " "
        elif stage == 'validation' or stage == "test":
            color = self.color_test_plt
            desc = f"Validating epoch {epoch+1}".ljust(desc_length) + " "
        else:
            color = self.color_test_plt
            desc = f"Making predictions"
        
        # Create the progress_bar object
        progress = tqdm(enumerate(dataloader), total=total, colour=color) #, dynamic_ncols=True
        progress.set_description(desc)

        return progress

    # Function that switchs training dataloaders based on selected stategy
    def _switch_dataloaders(
        self,
        epoch
        ):

        """
        Selects the appropriate training dataloader for the current epoch.

        Uses the standard dataloader for most epochs, and switches to the
        no-augmentation dataloader during the last `augmentation_off_epochs`
        to stabilize training (if specified).

        Args:
            epoch (int): Current epoch number (zero-indexed).

        Returns:
            torch.utils.data.DataLoader: The dataloader to use for this epoch.
        """

        # Strategy 1: Always use augmentation
        if self.augmentation_strategy == "always":
            return self.dataloaders['train']
        
        # Strategy 2: Disable augmentation for the last N epochs
        elif self.augmentation_strategy == "off_last":
            if self.augmentation_off_epochs > 0 and epoch >= (self.epochs - self.augmentation_off_epochs):
                if epoch == (self.epochs - self.augmentation_off_epochs):
                    self.info(f"Epoch {epoch+1}/{self.epochs}: Using no-augmentation dataloader for stabilization.")
                return self.dataloaders['train_aug_off']
            else:
                return self.dataloaders['train']
        
        # Strategy 3: Disable augmentation for the first N epochs
        elif self.augmentation_strategy == "off_first":
            if self.augmentation_off_epochs > 0 and epoch < self.augmentation_off_epochs:
                self.info(f"Epoch {epoch+1}/{self.epochs}: Using no-augmentation dataloader for stabilization.")
                return self.dataloaders['train_aug_off']
            else:
                return self.dataloaders['train']
        
        # Strategy 4: Randomly toggle augmentation per epoch
        elif self.augmentation_strategy == "random":
            use_aug = random.random() < self.augmentation_random_prob
            if not use_aug:                
                self.info(f"Epoch {epoch+1}/{self.epochs}: Using no-augmentation dataloader for stabilization.")
                return self.dataloaders['train_aug_off']                
            else:
                self.info(f"Epoch {epoch+1}/{self.epochs}: Using augmentation dataloader for stabilization.")
                return self.dataloaders['train']
            
    def _load_inference_model(
            self,
            model_state
    ):
        
        """
        LoadS the appropriate model for inference based on the given 'model_state'.
        
        Args:
            model_state (str): Specifies which model to load:
                - 'last' : last trained model
                - 'loss' : model with lowest validation loss                
                - int    : specific epoch number (1-based)
        
        Returns:
            model (torch.nn.Module): The model selected for inference.
        """
        
        if not (model_state in self.valid_modes or isinstance(model_state, int)):
            self.error(f"Invalid model value: {model_state}. Must be one of {self.valid_modes} or an integer.")

        # Handle named model states
        named_models = {
            "last": self.model,
            "loss": self.model_loss,
            "dice": self.model_dice,
            "iou": self.model_iou,            
        }

        if isinstance(model_state, int):
            if self.model_epoch is None or model_state > len(self.model_epoch):
                self.info(f"Model epoch {model_state} not found, using default model for prediction.")
                return self.model
            return self.model_epoch[model_state - 1]

        # Fallback for named models if they're not available
        model = named_models.get(model_state)
        if model is None:
            self.info(f"Model '{model_state}' not found, using last-epoch model for prediction.")
            model = self.model

        return model
    
    def _prepare_data(
            self,
            data
            ):

        """
        Prepare the data by performing the following operations:
        1. Moving images to the device (GPU/CPU).
        2. Moving tensor values within the targets dictionary to the device.
        
        Args:
            data:
                images (list): List of image tensors.
                targets (list of dict): List of dictionaries, where each dictionary contains target data, 
                                        potentially including tensor values.
        
        Returns:
            tuple: A tuple containing:
                - images (list): List of image tensors moved to the device.
                - targets (list of dict): List of dictionaries, with tensor values moved to the device.
        """
        images, targets = data
        if isinstance(targets, dict):
            targets = [targets]
        images = list(image.to(self.device) for image in images)            
        targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        return images, targets


    # ======================================= #
    # ============ VISUALIZATION ============ #
    # ======================================= #

    # Function to display and plot the results
    def _display_results(self):
    
        """
        Displays the training and validation results both numerically and visually.

        Functionality:
            Outputs key metrics such as training and validation loss, R2, and distributions
            Generates plots that visualize the training process, such as:
                - Loss curves (training vs validation loss over epochs).
                - R2 curves (training vs validation R2 over epochs).
                - Output distributions
        """
        
        # Clear output
        clear_output(wait=True)

        # Print results   
        print(self._format_epoch_results("train"))        
        if self.apply_validation:
            print(self._format_epoch_results("test"))
        
        # Plot training and test loss, R2, and distribution curves.
        if self.plot_curves:
            
            # First figure (top)
            n_cols = 3
            plt.figure(figsize=(25, 6), facecolor=self.figure_color_map['bg'])
            curr_epoch = len(self.results["train_loss"])
            range_epochs = range(1, curr_epoch+1)            

            # Plot loss
            ax = plt.subplot(1, n_cols, 1)
            self._plot(ax, range_epochs, 'loss')

            # Plot dice
            ax = plt.subplot(1, n_cols, 2)
            self._plot(ax, range_epochs, 'dice')
                    
            # Plot IoU
            ax = plt.subplot(1, n_cols, 3)
            #self._plot(ax, range_epochs, 'kde_train')
            self._plot(ax, range_epochs, 'iou')

            # Display plots
            plt.show()

            # Second figure (bottom)
            n_cols = 2
            plt.figure(figsize=(25, 6), facecolor=self.figure_color_map['bg'])
                                    
            # Plot time
            ax = plt.subplot(1, n_cols, 1)
            self._plot(ax, range_epochs, 'time')

            # Plot LR
            ax = plt.subplot(1, n_cols, 2)
            self._plot(ax, range_epochs, 'lr')           

            # Display plots
            plt.show()


    # ======================================= #
    # ===== INTERNAL CORE TRAINING LOGIC ==== #
    # ======================================= #

    # Fuction that initializes training process and validates training parameters
    def _init_train(
        self,
        target_dir: str=None,
        model_name: str=None,        
        enable_resume: bool=True,
        dataloaders: dict[str, torch.utils.data.DataLoader] = None,
        apply_validation: bool=True,
        augmentation_strategy: str="always",
        augmentation_off_epochs: int=5,
        augmentation_random_prob: float=0.5,
        save_best_model: Union[str, List[str]] = "last",  # Allow both string and list
        keep_best_models_in_memory: bool=False,        
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
            enable_resume (bool, optional): Enables resuming training from the last checkpoint. Default is True.
                - True: Training will resume from the most recent saved checkpoint. Useful if training is interrupted.
                - False: Checkpoints will not be saved, so training cannot be resumed after interruption. This speed up training.
            dataloaders (dict[str, torch.utils.data.DataLoader]): A dictionary containing a dataloader for training the model (mandatory), a dataloader for testing/validating the model (optional), and a dataloader without augmentation (optional).                                                
            apply_validation (bool, optional): Whether to apply validation after each epoch. Default is True.
            augmentation_strategy (str, optional): Determines how data augmentation is applied during training.
                - "always": augmentation is applied every epoch.
                - "off_last": augmentation is disabled during the last `augmentation_off_epochs` epochs.
                - "off_first": augmentation is disabled during the first `augmentation_off_epochs` epochs.
                - "random": augmentation is applied randomly according to `augmentation_random_prob`.
                Default is "always".
            augmentation_off_epochs (int, optional): Number of final epochs in which augmentation is disabled if `augmentation_strategy` is set to "off_last_n". Default is 5.
            augmentation_random_prob (float, optional): Probability (0.0-1.0) of applying augmentation in each batch if `augmentation_strategy` is set to "random". Default is 0.5.            
            epochs (int, optional): Number of epochs to train. Must be an integer greater than or equal to 1. Default is 30.
            plot_curves (bool, optional): Whether to plot training and validation curves. Default is True.
            amp (bool, optional): Enable automatic mixed precision for faster training. Default is True.
            enable_clipping (bool, optional): Whether to enable gradient clipping. Default is True.
            accumulation_steps (int, optional): Steps for gradient accumulation. Must be an integer greater than or equal to 1. Default is 1.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.
            save_best_model (Union[str, List[str]]): Criterion mode for saving the model: 
                - "loss": saves the epoch with the lowest validation loss                
                - "last": saves last epoch
                - "all": saves models for all epochs
                - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
                - None: the model will not be saved.

        Functionality:
            Validates input arguments with assertions.
            Prints configuration parameters using the `_print_config` method.
            Initializes the optimizer, loss function, and scheduler.
            Ensures the target directory for saving models exists, creating it if necessary.
            Sets the model name for saving, defaulting to the model's class name if not provided.
            Initializes structures to track the best-performing model and epoch-specific models:
            
        This method sets up the environment for training, ensuring all necessary resources and parameters are prepared.
        """

        self.info(f"Checking arguments...")

        # Validate resume
        if not isinstance(enable_resume, (bool)):
            self.error("'enable_resume' must be True or False.")
        else:
            self.resume = enable_resume
            if not self.resume:
                self.warning("'enable_resume' is set to False. If training interrupts, the last checkpoint will not be recovered.")
        
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
        else:
            self.accumulation_steps = accumulation_steps

        # Validate epochs
        if not isinstance(epochs, int) or epochs is None or epochs < 1:
            self.error(f"'epochs' must be an integer greater than or equal to 1.")
        self.epochs = epochs

        # Validate augmentation_strategy
        valid_aug_strategies = {"always", "off_last", "off_first", "random"}
        if augmentation_strategy not in valid_aug_strategies:
            self.error(f"'augmentation_strategy' must be any of these strings: {valid_aug_strategies}.")
        self.augmentation_strategy = augmentation_strategy

        # Validate augmentation_off_epochs
        if not isinstance(augmentation_off_epochs, (int)):
            self.error(f"'augmentation_off_epochs' must be a positive integer.")
        self.augmentation_off_epochs = min(augmentation_off_epochs, epochs)

        # Validate augmentation_random_prob
        if not isinstance(augmentation_random_prob, (int, float)) or not (0.0 <= float(augmentation_random_prob) <= 1.0):
            self.error(f"'augmentation_random_prob' must be a float between 0.0 and 1.0.")
        self.augmentation_random_prob = augmentation_random_prob

        # Validate plot_curves
        if not isinstance(plot_curves, (bool)):
            self.error(f"'plot_curves' must be True or False")
        else:
            self.plot_curves = plot_curves
        
        # Validate amp
        if not isinstance(amp, (bool)):
            self.error(f"'amp' must be True or False")
        else:
            self.amp = amp

        # Validate enable_clipping
        if not isinstance(enable_clipping, (bool)):
            self.error(f"'enable_clipping' must be True or False")
        else:
            self.enable_clipping = enable_clipping

        # Validate debug_mode
        if not isinstance(debug_mode, (bool)):
            self.error(f"'debug_mode' must be True or False")
        else:
            self.debug_mode = debug_mode
            
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
        if self.save_best_model:
            if not isinstance(mode, list):
                self.error(f"'mode' must be a string or a list of strings.")
            for m in mode:
                if m not in self.valid_modes:
                    self.error(f"Invalid mode value: '{m}'. Must be one of {self.valid_modes}")

        # Assign the validated mode list
        self.mode = mode

        # Assign dataloaders (to be validated later on)
        self.dataloaders = dataloaders
        
        # Initialize model name path and ensure target directory exists
        self.target_dir = target_dir or "outputs"
        os.makedirs(self.target_dir, exist_ok=True)  
        self.model_name = model_name or "unnamed.pth"

        # List of acceptable extensions
        valid_extensions = ['.pth', '.pt', '.pkl', '.h5', '.torch']

        # Check if model_name already has a valid extension, otherwise add the default .pth extension
        if not any(self.model_name.endswith(ext) for ext in valid_extensions):
            self.model_name += '.pth'

        # Initialize optimizer
        if self.optimizer is None:
            self.error("Invalid 'optimizer'. Some examples: torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD.")
        
        # Initialize loss_fn
        if self.loss_fn is None:
            self.error("Invalid 'loss_fn'. Some examples:  DiceCrossEntropyLoss().")

        # No need to check scheduler, it can be None

        # Initialize the display showing the numeric results
        self._init_results()

        # Default checkpoint path
        self.checkpoint_path = os.path.join(
            self.target_dir,
            f"{self.checkpoint_path_prefix}_{self.model_name}.gz"
        )

        # Remove leftover temp file from previous failed save
        tmp_path = self.checkpoint_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            self.info(f"Removed stale temp checkpoint file: {tmp_path}")

        # Initialize epoch number    
        self.start_epoch = 0

        # Load checkpoint if resume is enabled. This overrides the arguments                      
        if self.resume:

            # Collect all matching checkpoints recursively (newest first)
            checkpoints = sorted(
                Path(".").rglob(f"{self.checkpoint_path_prefix}*.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            if not checkpoints or (not os.path.isfile(self.checkpoint_path) and model_name is not None):
                #self.warning(f"'resume' enabled but {self.checkpoint_path} does not exist. Disabling 'resume' and starting training from epoch 1.")
                #self.resume = False        
                self.start_epoch, resume_msg = 0, ""
            else:                

                # Build candidate list: default path first, then all found checkpoints
                candidates = [Path(self.checkpoint_path)] + checkpoints
                
                # Loop over all detected checkpoints
                loaded = False
                self.info(f"Loading checkpoint...")
                for ckpt in candidates:
                    try:
                        self.checkpoint_path = str(ckpt)
                        self.start_epoch, resume_msg, ckpt_match = self._load_checkpoint()
                        if ckpt_match:
                            loaded = True                            
                            # self.info(f"Loaded {self.checkpoint_path}!.")
                            break
                    except Exception as e:                        
                        pass

                if not loaded:
                    self.error("No checkpoint could be loaded.")
        else:
            self.start_epoch, resume_msg = 0, ""

        # Check loss_fn        
        try:
            # Get number of parameters of the forward method
            num_args = len(inspect.signature(self.loss_fn.forward).parameters)
        except (AttributeError, TypeError) as e:
            self.error(f"Error inspecting loss function: {e}. "
                       "Perhaps the loss_fn is not correctly instantiated.")        
        expected_args = 2
        if num_args != expected_args:            
            self.error(f"Unexpected number of arguments in loss_fn.forward: {num_args}. ")
            
        # Validate fields in the dataloaders
        self._validate_dataloaders()
        
        # Get batch size from dataloaders
        if hasattr(self.dataloaders['train'], 'batch_size'):
            batch_size = self.dataloaders['train'].batch_size
        else:
            self.warning("Parameter 'batch_size' does not exist in the dataloader. Set to 4.")
            batch_size = 4  # or set a default value

        # Print configuration parameters
        self._print_config(batch_size=batch_size, resume_msg=resume_msg)

        self.info(f"Checking dataloaders...")   

        # Set the model in train mode for checking dataloaders
        self.model.eval()

        # Initialize num_classes
        self.num_classes = 0

        # Load the first image of the dataset for verification, and initialize results dict
        data = self.dataloaders['train'].dataset[0]
        if isinstance(data, (tuple, list)) and len(data) == 2:    
            X = data[0].unsqueeze(0).to(self.device) # Add batch dimension                
        else:
            self.error('The training dataset should contain two elements: image, label.')

        try:

            # Here is where the model will "complain" if the shape is incorrect
            with torch.no_grad():

                # We just try the first signal of the batch
                y_pred = self.get_predictions(self.model(X))

        except Exception as e:

            # Catch any unexpected errors (not shape-related)
            self.error(f"Unexpected error during input shape compatibility checking: {str(e)}")

        finally:
            
            # Always attempt to detect the number of classes
            try:

                # If y_pred was successfully computed, get the number of classes
                if 'y_pred' in locals():
                    self.num_classes = y_pred.shape[1]
                    self.info(f"Detected number of classes: {self.num_classes}.")                        
                
                # Otherwise, throw an exception
                else:
                    self.error("y_pred is not defined due to a failure in the forward pass.")
            
            except Exception as e:

                # Throw an exception if the shape of the output tensor is unexpected
                shape_info = getattr(y_pred, 'shape', 'undefined')
                self.error(f"Unexpected shape in of the prediction output: {shape_info}.")
        
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
            self.best_test_dice = float("-inf")
            self.best_test_iou = float("-inf")

        # Display last checkpoint results
        if self.resume == True and self.start_epoch > 0:                        
            self._display_results()
            self.info(f"Verification complete! Training continues.")
        else:
            self.info(f"Verification complete! Training beggings.")
    
    # Training step for a single epoch
    def _train_step(
        self,
        epoch: int = 1,
        ):
    
        """
        Trains a PyTorch model for a single epoch with gradient accumulation.

        Args:
            epoch: Epoch number.
        """

        # Measure time
        start_time = time.time()

        # Switch dataloaders according to 'augmentation_strategy'
        dataloader = self._switch_dataloaders(epoch)

        # Put model in train mode
        self.model.train()

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if self.amp else None

        # Setup train loss and train accuracy values
        len_dataloader = len(dataloader)
        train_loss, train_dice, train_iou = 0, 0, 0        

        # Loop through data loader data batches
        self.optimizer.zero_grad()  # Clear gradients before starting        
        for batch, (X, y) in self._progress_bar(
            dataloader=dataloader,
            total=len_dataloader,
            epoch=epoch,
            stage='train'
            ):
            
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)                

            # Optimize training with amp if available
            if self.amp:
                with autocast(device_type='cuda', dtype=torch.float16):

                    # Forward pass
                    y_pred = self.get_predictions(self.model(X))
                    
                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if self.enable_clipping:
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
                    loss = self.loss_fn(y_pred, y) / self.accumulation_steps

                    # Check for NaN or Inf in loss
                    if self.debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        self.warning(f"Loss is NaN or Inf at batch {batch}. Skipping...")
                        continue

                # Backward pass with scaled gradients
                if self.debug_mode:
                    # Use anomaly detection
                    with torch.autograd.detect_anomaly():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

            else:
                # Forward pass
                y_pred = self.get_predictions(self.model(X))
                
                # Calculate loss, normalize by accumulation steps
                loss = self.loss_fn(y_pred, y) / self.accumulation_steps

                # Backward pass
                loss.backward()

            # Gradient cliping
            if self.enable_clipping:
                # Apply clipping if needed
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Perform optimizer step and clear gradients every accumulation_steps
            if (batch + 1) % self.accumulation_steps == 0 or (batch + 1) == len_dataloader:

                if self.amp:

                    # Gradient cliping
                    if self.enable_clipping:
                        # Unscale the gradients before performing any operations on them
                        scaler.unscale_(self.optimizer)
                        # Apply clipping if needed
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Check gradients for NaN or Inf values
                    if self.debug_mode:
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

                else:
                    # Optimizer step
                    self.optimizer.step()

                # Optimizer zero grad
                self.optimizer.zero_grad()

            # Accumulate metrics
            train_loss += loss.item() * self.accumulation_steps  # Scale back to original loss
            y_pred = y_pred.float() # Convert to float for stability
            train_dice += self.calculate_mask_similarity(y, y_pred, metric='dice', num_classes=self.num_classes) # This returns a cpu scalar
            train_iou +=  self.calculate_mask_similarity(y, y_pred, metric='iou', num_classes=self.num_classes) # This returns a cpu scalar
    
        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len_dataloader
        train_dice = train_dice / len_dataloader
        train_iou = train_iou / len_dataloader

        # Clear local variables        
        self.clear_cuda_memory(['X', 'y', 'y_pred', 'loss'], locals())
        
        # Compute elapsed time 
        elapsed_time = time.time() - start_time

        # Retrieve the learning rate
        if self.scheduler is None or isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr = self.optimizer.param_groups[0]['lr']
        else:
            lr = self.scheduler.get_last_lr()[0]
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)
        self.results["train_loss"].append(train_loss)
        self.results["train_dice"].append(train_dice)
        self.results["train_iou"].append(train_iou)        
        self.results["train_time [s]"].append(elapsed_time)
        self.results["lr"].append(lr)

    # Validation/test step for a single epoch
    def _test_step(
        self,
        epoch: int = 1,
        ):
        
        """
        Tests a PyTorch model for a single epoch.

        Args:
            epoch: Epoch number.
        """

        # Execute the test step is apply_validation is enabled
        if self.apply_validation:

            # Measure time
            start_time = time.time()

            # Get test dataloader
            dataloader = self.dataloaders['test']

            # Put model in eval mode
            self.model.eval()            
            
            # Setup test loss and test R2 values
            len_dataloader = len(dataloader)
            test_loss, test_dice, test_iou = 0, 0, 0            

            # Set inference context
            try:
                inference_context = torch.inference_mode()
                with torch.inference_mode():
                    # Load the first image and add the batch dimension
                    data = dataloader.dataset[0]
                    X = data[0].unsqueeze(0)                    
                    y_pred = self.get_predictions(self.model(X.to(self.device)))
                    
            except RuntimeError:
                inference_context = torch.no_grad()
                #self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

            # Turn on inference context manager 
            with inference_context:
                
                # Loop through DataLoader batches
                for batch, (X, y) in self._progress_bar(
                    dataloader=dataloader,
                    total=len_dataloader,
                    epoch=epoch,
                    stage='test'):

                    # Send data to target device
                    X, y = X.to(self.device), y.to(self.device)                        
                    
                    if torch.isnan(X).any() or torch.isinf(X).any():
                        self.warning(f"NaN or Inf detected in test input!")

                    # Enable AMP if specified
                    with torch.autocast(device_type='cuda', dtype=torch.float16) if self.amp else nullcontext():

                        # Forward pass
                        y_pred = self.get_predictions(self.model(X))

                        # Check for NaN/Inf in predictions
                        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                            if self.enable_clipping:
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
                        if self.debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                            self.warning(f"Loss is NaN/Inf at batch {batch}. Skipping...")
                            continue

                    # Calculate and accumulate metrics: dice, iou
                    y_pred = y_pred.float() # Convert to float for stability
                    test_dice += self.calculate_mask_similarity(y, y_pred, metric='dice', num_classes=self.num_classes) # This returns a cpu scalar
                    test_iou +=  self.calculate_mask_similarity(y, y_pred, metric='iou', num_classes=self.num_classes) # This returns a cpu scalar
                    
            
            # Adjust metrics to get average loss and accuracy per batch 
            test_loss = test_loss / len_dataloader
            test_dice = test_dice / len_dataloader
            test_iou = test_iou / len_dataloader

            # Remove variables
            self.clear_cuda_memory(['X', 'y', 'y_pred', 'loss'], locals())            
            
            # Compute elapsed time 
            elapsed_time = time.time() - start_time
        
        # Otherwise set params with initial values
        else:
            test_loss, test_dice, test_iou = self.best_test_loss, self.best_test_dice, self.best_test_iou
            elapsed_time = 0.0

        # Scheduler step after the optimizer
        self._scheduler_step(
            test_loss=test_loss,
            test_dice=test_dice
            )
        
        # Update results dictionary
        self.results["test_loss"].append(test_loss)        
        self.results["test_dice"].append(test_dice)
        self.results["test_iou"].append(test_iou)
        self.results["test_time [s]"].append(elapsed_time)        
    
    # Scheduler step after the optimizer
    def _scheduler_step(
        self,
        test_loss: float=None,
        test_dice: float=None,
        ):

        """
        Performs a scheduler step after the optimizer step.

        Args:            
            test_loss (float, optional): Test loss value, required for ReduceLROnPlateau with 'min' mode.
            test_acc (float, optional): Test accuracy value, required for ReduceLROnPlateau with 'max' mode.
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
                        f"The scheduler requires either `test_loss` or `test_acc` "
                        "depending on its mode ('min' or 'max')."
                        )
            else:
                self.scheduler.step()  # For other schedulers

    # Updates and saves the best model and model_epoch list based on the specified mode.
    def _update_model(self) -> pd.DataFrame:

        """
        Updates and saves the best model and model_epoch list based on the specified mode(s), as well as the last-epoch model.

        Returns:
            A dataframe of training and testing metrics for each epoch.
    
        Functionality:
            Saves the last-epoch model.
            Saves the logs (self.results).
            Saves the best-performing model during training based on the specified evaluation mode.
            If mode is "all", saves the model for every epoch.
            Updates `self.model_<loss, acc, fpr, pauc, epoch>` accordingly.
        """

        if isinstance(self.mode, str):
            self.mode = [self.mode]  # Ensure self.mode is always a list

        # Get actual epoch number        
        epoch = self.results["epoch"][-1]

        # If 'apply_validation' is enabled, then update models based on validation results
        if self.apply_validation:
            test_loss = self.results["test_loss"][-1]
            test_dice = self.results["test_dice"][-1]
            test_iou = self.results["test_iou"][-1]            
        else:
            test_loss = self.results["train_loss"][-1]
            test_dice = self.results["train_dice"][-1]
            test_iou = self.results["train_iou"][-1]            

        # Some helper functions
        def remove_previous_best(model_name):

            """
            Removes previously saved best model files.
            """

            file_to_remove = glob.glob(os.path.join(self.target_dir, model_name.replace(".", "_epoch*.")))
            for f in file_to_remove:
                os.remove(f)

        def save_model(model_name):

            """
            Helper function to save the model.
            """

            #self.save(model=self.model, target_dir=self.target_dir, model_name=model_name.replace(".", f"_epoch{epoch+1}."))
            self.save(model=self.model, target_dir=self.target_dir, model_name=model_name.replace(".", f"_epoch{epoch}."))
        
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

    def _finish_train(
        self,
        elapsed_time: float=None,
        ):

        """
        Finalizes the training process by showing the elapsed time.
        
        Args:
            elapsed_time: Elapsed time.
        """

        # Remove checkpoint
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            #self.info(f"Removed {self.checkpoint_path}.")

        # Print elapsed time
        self.info(f"Training finished! Elapsed time: {self.sec_to_min_sec(elapsed_time)}")

    def _save_checkpoint(
        self,
        next_epoch
        ):

        """
        Safely saves the current training state (model weights, optimizer, scheduler, etc.)
        to a compressed checkpoint file.

        Args:
            next_epoch (int): The epoch number you will start from next time (usually current_epoch + 1).

        This creates a compressed file 'checkpoint.pth.gz' containing:
            - model state_dict            
            - optimizer state_dict
            - scheduler state_dict
            - engine internal state (device, mode, results, etc.)
        """

        if self.resume:            

            # Define paths
            final_path = self.checkpoint_path           # e.g., 'checkpoint.pth.gz'
            temp_path = self.checkpoint_path + ".tmp"   # e.g., 'checkpoint.pth.gz.tmp'
            backup_path = self.checkpoint_path + "_bk"  # e.g., 'checkpoint.pth.gz_bk'

            checkpoint = {
                'model_state': self.model.state_dict(),            
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,            
                'loss_fn': self.loss_fn, #Warning: this checkpoint will not work if the class is renamed, removed, or moved.
                'checkpoint_path': self.checkpoint_path,
                'next_epoch': next_epoch,                        
                'engine_state': {                
                    'target_dir': self.target_dir,
                    'device': self.device,
                    'accumulation_steps': self.accumulation_steps,                
                    'augmentation_strategy': self.augmentation_strategy,
                    'augmentation_off_epochs': self.augmentation_off_epochs,
                    'augmentation_random_prob': self.augmentation_random_prob,                    
                    #'dataloaders': self.dataloaders,
                    'debug_mode': self.debug_mode,                
                    'amp': self.amp,
                    'enable_clipping': self.enable_clipping,
                    'keep_best_models_in_memory': self.keep_best_models_in_memory,
                    'log_verbose': self.log_verbose,
                    'mode': self.mode,
                    'model_name': self.model_name,
                    'num_epochs': self.epochs,
                    'plot_curves': self.plot_curves,                
                    'results': self.results,
                    'save_best_model': self.save_best_model,                 
                }
            }

            try:
                # Save new checkpoint to a temporary file
                with gzip.open(temp_path, 'wb') as f:
                    torch.save(checkpoint, f)

                # Move current checkpoint to backup (if exists)
                if os.path.exists(final_path):
                    os.replace(final_path, backup_path)

                # Replace final path with the new checkpoint
                os.replace(temp_path, final_path)

                # Delete backup if everything went fine
                if os.path.exists(backup_path):
                    os.remove(backup_path)

            except Exception as e:
                self.error(f"Failed to save checkpoint: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)  # Clean up temp file
                print.info("Original checkpoint is still safe.")
                raise
        
            #finally:
                #self.info(f"Saved {self.checkpoint_path} to resume training later.")

    def _load_checkpoint(self):

        """
        Loads the training state from the last saved checkpoint.

        Returns:
            int: The epoch number loaded from the checkpoint (start from this epoch when resuming).

        Behavior:
            - Looks for 'checkpoint.pth.gz' first (compressed file)
            - Falls back to 'checkpoint.pth' if needed (uncompressed)
            - Restores model, optimizer, scheduler,
            and engine internal state.
        """
        
        start_epoch = 0
        resume_msg = ""
        ckpt_match = False
            
        # Check if the checkpoint file exists
        if self.resume and os.path.isfile(self.checkpoint_path):            
            
            # Load checkpoint
            #from engines.loss_functions
            with gzip.open(f"{self.checkpoint_path}", 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)            
            if os.path.basename(checkpoint['checkpoint_path']) != os.path.basename(self.checkpoint_path):
                return start_epoch, resume_msg, ckpt_match
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state'])
            
            # Load optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Load scheduler (it can be None)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            # Load loss_fn
            self.loss_fn = checkpoint.get('loss_fn', None) #loss_functions.DiceCrossEntropyLoss(num_classes=self.num_classes, label_smoothing=0.1))
            
            # Restore engine internal state
            engine_state = checkpoint.get('engine_state', {})           
            self.target_dir = engine_state.get('target_dir', 'models')            
            self.device = engine_state.get('device', self.device)            
            self.accumulation_steps = engine_state.get('accumulation_steps', 1)            
            self.augmentation_strategy = engine_state.get('augmentation_strategy', "always")            
            self.augmentation_off_epochs = engine_state.get('augmentation_off_epochs', 5)            
            self.augmentation_random_prob = engine_state.get('augmentation_random_prob', 0.5)            
            #self.dataloaders = engine_state.get('dataloaders', None)            
            self.debug_mode = engine_state.get('debug_mode', False)            
            self.amp = engine_state.get('amp', True)            
            self.enable_clipping = engine_state.get('enable_clipping', True)            
            self.keep_best_models_in_memory = engine_state.get('keep_best_models_in_memory', False)            
            self.log_verbose = engine_state.get('log_verbose', True)            
            self.mode = engine_state.get('mode', "last")            
            self.model_name = engine_state.get('model_name', 'model')                        
            self.epochs = engine_state.get('num_epochs', 30)            
            self.plot_curves = engine_state.get('plot_curves', True)            
            self.results = engine_state.get('results', {})            
            self.save_best_model = engine_state.get('save_best_model', True)            
            
            # Return the epoch to resume from
            start_epoch = checkpoint.get('next_epoch', 0)
            
            # Print successful loading
            resume_msg = f"sucessfully loaded checkpoint from epoch {start_epoch}"
            ckpt_match = True

        return start_epoch, resume_msg, ckpt_match

           
    # ======================================= #
    # ============= PUBLIC API ============== #
    # ======================================= #

    def save(
        self,
        model: torch.nn.Module,
        target_dir: str,
        model_name: str
        ):

        """
        Saves a PyTorch model to a target directory.

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
        model_name: str,        
        export: bool = False
        ):
        
        """
        Loads a PyTorch model from a target directory and optionally returns it.

        Args:
            target_dir (str): A directory where the model is located.
            model_name (str): The name of the model to load. Should include:            
                ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.
            export (bool, optional): If True, returns the loaded model instead of just loading it. Default is False.

        Returns:
            The loaded PyTorch model.
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
    
        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(state_dict)
        if export:
            return self.model
        
        return self

    # Trains and tests a Pytorch model
    def train(
        self,
        target_dir: str=None,
        model_name: str=None,
        enable_resume: bool=True,
        dataloaders: dict[str, torch.utils.data.DataLoader]=None, 
        save_best_model: Union[str, List[str]] = "last",
        keep_best_models_in_memory: bool=False,                
        apply_validation: bool=True,        
        augmentation_strategy: str="always",
        augmentation_off_epochs: int=5,
        augmentation_random_prob: float=0.5,        
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
            enable_resume (bool, optional): Enables resuming training from the last checkpoint. Default is True.
                - True: Training will resume from the most recent saved checkpoint. Useful if training is interrupted.
                - False: Checkpoints will not be saved, so training cannot be resumed after interruption. This speed up training.
            dataloaders (dict[str, torch.utils.data.DataLoader]): A dictionary containing a dataloader for training the model (mandatory), a dataloader for testing/validating the model (optional), and a dataloader without augmentation (optional).            
            save_best_model (Union[str, List[str]], optional): Criterion(s) for saving the model.
                Options include:
                - "loss" (validation loss),
                - "dice" (validation Dice-coefficient),
                - "iou" (validation IoU),                
                - "last" (save model at the last epoch),
                - "all" (save models for all epochs),
                - A list, e.g., ["loss", "fpr"], is also allowed. Only applicable if `save_best_model` is True.
            keep_best_models_in_memory (bool, optional): If True, the best models are kept in memory for future inference. The model state from the last epoch will always be kept in memory.            
            apply_validation (bool, optional): Whether to apply validation after each epoch. Default is True.
            augmentation_strategy (str, optional): Determines how data augmentation is applied during training.
                - "always": augmentation is applied every epoch.
                - "off_last": augmentation is disabled during the last `augmentation_off_epochs` epochs.
                - "off_first": augmentation is disabled during the first `augmentation_off_epochs` epochs.
                - "random": augmentation is applied randomly according to `augmentation_random_prob`.
                Default is "always".
            augmentation_off_epochs (int, optional): Number of final epochs in which augmentation is disabled if `augmentation_strategy` is set to "off_last". Default is 5.
            augmentation_random_prob (float, optional): Probability (0.0-1.0) of applying augmentation in each batch if `augmentation_strategy` is set to "random". Default is 0.5.                        
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
        start_time = time.time()

        # Initialize training process and check arguments
        self._init_train(
            target_dir=target_dir,
            model_name=model_name,            
            enable_resume=enable_resume,
            dataloaders=dataloaders,
            save_best_model=save_best_model,
            keep_best_models_in_memory=keep_best_models_in_memory,
            apply_validation=apply_validation,
            augmentation_strategy=augmentation_strategy,
            augmentation_off_epochs=augmentation_off_epochs,
            augmentation_random_prob=augmentation_random_prob,            
            epochs=epochs, 
            plot_curves=plot_curves,
            amp=amp,
            enable_clipping=enable_clipping,
            accumulation_steps=accumulation_steps,
            debug_mode=debug_mode,
            )
            
        # Loop through training and testing steps for a number of epochs
        # If 'resume' is True, resume training from 'self.start_epoch', the checkpoint epoch, otherwise self.start_epoch = 0
        # 'self.epochs' is the total number of epochs originally set
        for epoch in range(self.start_epoch, self.epochs):

            # Perform training step  
            self._train_step(epoch)            

            # Perform test step
            self._test_step(epoch)            
            
            # Show results
            self._display_results()

            # Update and save the best model, model_epoch list based on the specified mode, and the actual-epoch model.            
            df_results = self._update_model()
            
            # Save a checkpoint to allow resuming training later            
            self._save_checkpoint(next_epoch = epoch + 1)

        # Finish training process
        total_elapsed_time = time.time() - start_time
        self._finish_train(total_elapsed_time)

        return df_results

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_classes: int=2,
        model_state: str="last",
        output_type: str="onehot",
        ) -> torch.Tensor:

        """
        Predicts classes for a given dataset using a trained model.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataset to predict on.
            model_state: specifies the model to use for making predictions. "loss", "acc", "fpr", "pauc", "last" (default), "all", an integer
            output_type (str): The type of output to return. Either "softmax", "logits", or "argmax".            

        Returns:
            (list): All of the predicted class labels represented by prediction probabilities (softmax)
        """

        self.info(f"Checking arguments...")

        # Select model to use
        model = self._load_inference_model(model_state)          

        # Check output_max
        valid_output_types = {"softmax", "argmax", "onehot"}
        if output_type not in valid_output_types:
            self.error(f"Invalid output_max value: {output_type}. Must be one of {valid_output_types}.")
        
        # Initialize output list and prepare the model to work in inference mode
        y_preds = []
        model.eval()
        model.to(self.device)

        # Set batch size
        batch_size = dataloader.batch_size
        self.num_classes = num_classes

        self.info(f"Checking dataloader...")

        # Set inference context
        try:
            inference_context = torch.inference_mode()
            with torch.inference_mode():                                
                # Load the first image
                data = dataloader.dataset[0]
                X = data[0].unsqueeze(0) # Add batch dimension to the image                
                check = self.get_predictions(model(X.to(self.device)))
            
        except RuntimeError:
            inference_context = torch.no_grad()
            #self.warning(f"torch.inference_mode() check caused an issue. Falling back to torch.no_grad().")

        # Free up unused GPU memory after shape-checking
        torch.cuda.empty_cache()

        self.info(f"Verification complete! Predition beggins.")

        # Turn on inference context manager 
        with inference_context:
            for batch, (X, y) in self._progress_bar(
                dataloader=dataloader,
                total=len(dataloader),
                stage='inference'
                ):

                # Send data and targets to target device
                X, y = X.to(self.device), y.to(self.device)
                
                # Do the forward pass
                if output_type == "softmax":
                    y_pred = torch.argmax(model(X), dim=1)
                elif output_type == "onehot":
                    y_pred = F.one_hot(
                        torch.argmax(model(X), dim=1),
                        num_classes=num_classes
                        ).permute(0, 3, 1, 2).float()  # Shape (B, C, H, W)
                
                if y_pred.shape[0] != batch_size:
            
                    # Pad with zeros along the channel dimension if there are fewer channelS
                    pad_size = batch_size - y_pred.shape[0]
                    padding = torch.zeros(pad_size, y_pred.shape[1], y_pred.shape[2], y_pred.shape[3], device=self.device)
                    y_pred = torch.cat((y_pred, padding), dim=0)
                
                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.unsqueeze(0).cpu())

        # Concatenate list of predictions into a tensor
        return torch.cat(y_preds, dim=0)