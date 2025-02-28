"""
Contains common classes for training and testing a PyTorch models.
"""

import torch
import numpy as np
import warnings
import sys
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

    @staticmethod
    def sec_to_min_sec(seconds):
        """Converts seconds to a formatted string in minutes and seconds."""
        if not isinstance(seconds, (int, float)) or seconds < 0:
            Logger().error("Input must be a non-negative number.")
                    
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        return f"{str(minutes).rjust(3)}m{str(remaining_seconds).zfill(2)}s"

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """Calculates accuracy between truth labels and predictions."""
        assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
        return torch.eq(y_true, y_pred).sum().item() / len(y_true)
    
    @staticmethod
    def calculate_f1_score(y_true, y_pred):
        """Calculates the F1 score between truth labels and predictions."""
        assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."

        # Convert tensors to binary values (assuming y_pred contains logits or probabilities)
        y_true = y_true.int()
        y_pred = y_pred.int()

        # True Positives, False Positives, False Negatives
        tp = torch.sum((y_true == 1) & (y_pred == 1)).item()
        fp = torch.sum((y_true == 0) & (y_pred == 1)).item()
        fn = torch.sum((y_true == 1) & (y_pred == 0)).item()

        # Compute Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Compute F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1


    @staticmethod
    def calculate_fpr_at_recall(y_true, y_pred_probs, recall_threshold):
        """Calculates the False Positive Rate (FPR) at a specified recall threshold."""
        if not (0 <= recall_threshold <= 1):
            Logger().error(f"'recall_threshold' must be between 0 and 1.")


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
        print(f"Saving best model to: {model_save_path}")
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
        Logger().info(f"Loading model from: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        
        return model
    
    @staticmethod
    def get_predictions(output):
        if isinstance(output, torch.Tensor):
            return output.contiguous()
        elif hasattr(output, "logits"):            
            return output.logits.contiguous()
        else:
            Logger().error(f"Unexpected model output type: {type(output)}")
