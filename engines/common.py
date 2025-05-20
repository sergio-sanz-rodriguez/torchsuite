"""
Contains common classes for training and testing a PyTorch models.
"""

import torch
import gc
import numpy as np
import warnings
import sys
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, auc

import warnings
warnings.filterwarnings("ignore")

class Colors:
    # Console escape codes
    BLACK = '\033[30m'
    BLUE = '\033[34m'
    ORANGE = '\033[38;5;214m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[38;2;255;215;0m' #'\033[33m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    LIGHT_GRAY = '\033[37;1m'
    DARK_GRAY = '\033[90m'
    LIGHT_BLUE = '\033[94m'
    LIGHT_GREEN = '\033[92m'
    LIGHT_RED = '\033[91m'
    LIGHT_YELLOW = '\033[93m'
    LIGHT_MAGENTA = '\033[95m'
    LIGHT_CYAN = '\033[96m'

    # Matplotlib compatible color names
    MATPLOTLIB_COLORS = {
        'black': '#000000',
        'blue': '#1f77b4',
        'orange': '#FFA500',
        'green': '#008000',
        'red': '#FF0000',
        'yellow': '#FFD700',
        'magenta': '#FF00FF',
        'cyan': '#00FFFF',
        'white': '#FFFFFF',
        'light_gray': '#D3D3D3',
        'dark_gray': '#A9A9A9',
        'light_blue': '#ADD8E6',
        'light_green': '#32CD32',
        'light_red': '#FF6347',
        'light_yellow': '#FFFACD',
        'light_magenta': '#EE82EE',
        'light_cyan': '#E0FFFF'
    	}

    @staticmethod
    def get_console_color(color_name):
        # Return console escape code
        return getattr(Colors, color_name.upper(), '')

    @staticmethod
    def get_matplotlib_color(color_name):
        # Return Matplotlib-compatible hex color
        return Colors.MATPLOTLIB_COLORS.get(color_name.lower(), '#000000')  # Default to black

# Logger class
class Logger:
    def __init__(self, log_verbose: bool=True):

        self.info_tag =    f"{Colors.GREEN}[INFO]{Colors.BLACK}"
        self.warning_tag = f"{Colors.ORANGE}[WARNING]{Colors.BLACK}"
        self.error_tag =   f"{Colors.RED}[ERROR]{Colors.BLACK}"
        self.log_verbose = log_verbose
    
    def info(self, message: str):
        print(f"{self.info_tag} {message}") if self.log_verbose else None
    
    def warning(self, message: str):
        print(f"{self.warning_tag} {message}", file=sys.stderr) if self.log_verbose else None
    
    def error(self, message: str):
        print(f"{self.error_tag} {message}", file=sys.stderr) if self.log_verbose else None
        raise ValueError(message)


# Common utility class
class Common(Logger):

    """A class containing utility functions for classification tasks."""
    def __init__(self, log_verbose: bool=True):
        super().__init__(log_verbose=log_verbose)

    def sec_to_min_sec(self, seconds):

        """
        Converts seconds to a formatted string in minutes and seconds.
        """

        if not isinstance(seconds, (int, float)) or seconds < 0:
            self.error("Input must be a non-negative number.")
            return None
                    
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        return f"{str(minutes).rjust(3)}m{str(remaining_seconds).zfill(2)}s"

    def calculate_accuracy(self, y_true, y_pred):

        """
        Calculates accuracy between truth labels and predictions.
        """

        if len(y_true) != len(y_pred):
            self.error(f"Length of y_true and y_pred for accuracy calculation must be the same.")

        return torch.eq(y_true, y_pred).sum().item() / len(y_true)
    
    def calculate_f1_score(self, y_true, y_pred, num_classes, average="macro"):

        """Calculates the F1 score for multi-class classification.

        Args:
            y_true (torch.Tensor): Ground truth labels (shape: [batch_size]).
            y_pred (torch.Tensor): Predicted labels (shape: [batch_size]).
            num_classes (int): Number of classes.
            average (str): 'macro' for macro-average, 'weighted' for weighted-average, 'micro' for micro-average.

        Returns:
            float: Computed F1 score.
        """

        if len(y_true) != len(y_pred):
            self.error(f"Length of y_true and y_pred for F1-score calculation must be the same.")

        # Convert tensors to integer labels
        y_true = y_true.int()
        y_pred = y_pred.int()
        f1_scores = []

        # Calculate metrics per class
        for class_idx in range(num_classes):
            tp = torch.sum((y_true == class_idx) & (y_pred == class_idx)).item()
            fp = torch.sum((y_true != class_idx) & (y_pred == class_idx)).item()
            fn = torch.sum((y_true == class_idx) & (y_pred != class_idx)).item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        f1_scores = torch.tensor(f1_scores, dtype=torch.float)

        # Calculate the average
        if average == "macro":
            return torch.mean(f1_scores).item()
        elif average == "weighted":
            class_counts = torch.tensor([(y_true == i).sum().item() for i in range(num_classes)], dtype=torch.float)
            total_samples = class_counts.sum()
            return (f1_scores * (class_counts / total_samples)).sum().item() if total_samples > 0 else 0
        elif average == "micro":
            total_tp = sum(torch.sum((y_true == i) & (y_pred == i)).item() for i in range(num_classes))
            total_fp = sum(torch.sum((y_true != i) & (y_pred == i)).item() for i in range(num_classes))
            total_fn = sum(torch.sum((y_true == i) & (y_pred != i)).item() for i in range(num_classes))

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        else:
            self.error("Invalid value for 'average'. Choose 'macro', 'weighted', or 'micro'.")

    def calculate_fpr_at_recall(self, y_true, y_pred_probs, recall_threshold):
        
        """
        Calculates the False Positive Rate (FPR) at a specified recall threshold.
        """

        if not (0 <= recall_threshold <= 1):
            self.error(f"'recall_threshold' must be between 0 and 1.")


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

    def calculate_pauc_at_recall(self, y_true, y_pred_probs, recall_threshold=0.80, num_classes=101):
        
        """
        Calculates the Partial AUC for multi-class classification at the given recall threshold.
        """

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


    def calculate_mask_similarity(self, y_true, y_pred, metric='dice', num_classes=4, from_logits=True, eps=1e-6, reduction='mean'):

        """
        Computes the Dice Coefficient for binary or multi-class segmentation.
        """

        # Check that the dimensions of y_true and y_pred are the same
        if len(y_true) != len(y_pred):
            self.error(f"Length of y_true and y_pred for Dice coefficient calculation must be the same.")
        
        # Ensure the inputs are 4D: [batch_size, channels, height, width]
        if y_true.dim() != 4:
            y_true = y_true.unsqueeze(0)
        
        if y_pred.dim() != 4:
            y_pred = y_pred.unsqueeze(0)

        # Apply softmax for multi-class segmentation
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1) if self.num_classes > 1 else torch.sigmoid(y_pred)

        # Ensure ground truth is float for stability
        y_true = y_true.float()

        # Binary case (num_classes == 1)
        if num_classes == 1:
            
            # Compute intersection and union
            intersection = torch.sum(y_true * y_pred, dim=(2, 3))
            if metric == 'dice':
                union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3))
            elif metric == 'iou':
                union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3)) - intersection

        # Multi-class case (num_classes > 1)
        else:
            # Convert predictions to one-hot-like representation
            y_pred_one_hot = F.one_hot(torch.argmax(y_pred, dim=1), num_classes=self.num_classes)
            y_pred_one_hot = y_pred_one_hot.permute(0, 3, 1, 2).float()  # Shape (B, C, H, W)

            # Compute intersection and union
            intersection = torch.sum(y_true * y_pred_one_hot, dim=(2, 3))
            if metric == 'dice':
                union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred_one_hot, dim=(2, 3))
            elif metric == 'iou':
                union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred_one_hot, dim=(2, 3)) - intersection

        # Compute metric
        if metric == 'dice':
            output = (2. * intersection + eps) / (union + eps)
        elif metric == 'iou':
            output = (intersection + eps) / (union + eps)
        if torch.any(output < 0) or torch.any(output > 1):
            self.warning(f"Mask similarity: {metric} coefficient out of range for class.")
        output = torch.clamp(output, 0.0, 1.0)
            
        # Reduce across classes
        if reduction == 'mean':
            return output.mean().item()
        elif reduction == 'sum':
            return output.sum().item()
        else:
            return output.cpu()
    
    def get_predictions(self, output):
        if isinstance(output, torch.Tensor):
            return output.contiguous()
        elif hasattr(output, "logits"):            
            return output.logits.contiguous()
        else:
            self.error(f"Unexpected model output type: {type(output)}")
    
    def clear_cuda_memory(self, var_names, namespace):
        """Clears listed variable names and empties CUDA cache."""
        if torch.cuda.is_available():
            for name in var_names:
                if name in namespace:
                    del namespace[name]
            gc.collect()
            torch.cuda.empty_cache()

    def save_model(self, model: torch.nn.Module, target_dir: str, model_name: str):

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
        self.info(f"Saving best model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)


    def load_model(self, model: torch.nn.Module, target_dir: str, model_name: str):
        
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
        self.info(f"Loading model from: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        
        return model

    def encode_ordinary_regression(self, label, num_classes=100):
        # label in range 1–100 → target vector of length 99
        return torch.FloatTensor([1 if i < label - 1 else 0 for i in range(num_classes - 1)])
    
    def decode_ordinary_regression(self, logits):
        # logits → sigmoid → binary decisions
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1) + 1

        