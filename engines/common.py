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

    def dice_coefficient_(self, y_true, y_pred, threshold=0.5, eps=1e-6):

        """
        Computes Dice Coefficient for binary or multi-class segmentation.
        """
        
        # Multiclass case, apply softmax
        if y_pred.dim() == 4 and y_pred.shape[1] > 1:
            y_pred = torch.argmax(y_pred, dim=1)  # Take the class with the highest probability
        
        # For binary case, apply sigmoid
        else:
            y_pred = torch.sigmoid(y_pred)

        # Convert predictions to binary
        y_pred = y_pred > threshold
        
        # Flatten the tensors
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        # Calculate the intersection and the union
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)

        # Compute the Dice coefficient
        dice = (2. * intersection + eps) / (union + eps)

        return dice


    def dice_coefficient(self, y_true, y_pred, num_classes=1, from_logits=True, eps=1e-6, reduction='mean'):
        
        """
        Computes the Dice Coefficient for binary or multi-class segmentation.
        """        

        # Check out lenghts
        if len(y_true) != len(y_pred):
            self.error(f"Length of y_true and y_pred for Dice coefficient calculation must be the same.")
        
        # Ensure 4K shape
        if y_true.dim() != 4:
            y_true.unsqueeze(0)
        
        if y_pred.dim() != 4:
            y_pred.unsqueeze(0)

        # Apply sigmoid if logits are provided
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        # Ensure ground truth is float for stability
        y_true = y_true.float()

        # For multi-class, treat each class as a binary mask
        if num_classes > 1:            
            y_pred = (y_pred > 0.5)

        # Compute the dice coefficient per class
        dice_scores = []
        for i in range(num_classes):
            pred_class = y_pred[:, i, :, :]
            true_class = y_true[:, i, :, :]

            intersection = torch.sum(true_class * pred_class)
            union = torch.sum(true_class) + torch.sum(pred_class)

            dice = (2. * intersection + eps) / (union + eps)
            dice_scores.append(dice)

        # Stack class-wise dice scores
        dice_scores = torch.stack(dice_scores)  

        # Reduce across classes
        if reduction == 'mean':
            return torch.mean(dice_scores).item()
        elif reduction == 'sum':
            return torch.sum(dice_scores).item()
        else:
            return dice_scores.cpu()

    def intersection_over_union(self, y_true, y_pred, num_classes=1, from_logits=True, eps=1e-6, reduction='mean'):
        
        """
        Computes the Intersection over Union (IoU) for binary or multi-class segmentation.
        The inputs should be tesors with shape (batch, channels/classes, height, width)
        """

        # Check out lenghts
        if len(y_true) != len(y_pred):
            self.error(f"Length of y_true and y_pred for IoU calculation must be the same.")
        
        # Ensure 4K shape
        if y_true.dim() != 4:
            y_true.unsqueeze(0)
        
        if y_pred.dim() != 4:
            y_pred.unsqueeze(0)

        # Apply sigmoid if logits are provided
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        # Ensure ground truth is float for stability
        y_true = y_true.float()

        # For multi-class, treat each class as a binary mask
        if num_classes > 1:
            y_pred = (y_pred > 0.5)

        # Compute the IoU score per class
        iou_scores = []
        for i in range(num_classes):
            pred_class = y_pred[:, i, :, :]
            true_class = y_true[:, i, :, :]

            intersection = torch.sum(true_class * pred_class)
            union = torch.sum(true_class) + torch.sum(pred_class) - intersection

            dice = (intersection + eps) / (union + eps)
            iou_scores.append(dice)

        # Stack class-wise iou scores
        iou_scores = torch.stack(iou_scores)  

        # Reduce across classes
        if reduction == 'mean':
            return torch.mean(iou_scores).item()
        elif reduction == 'sum':
            return torch.sum(iou_scores).item()
        else:
            return iou_scores.cpu()

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
    
    def get_predictions(self, output):
        if isinstance(output, torch.Tensor):
            return output.contiguous()
        elif hasattr(output, "logits"):            
            return output.logits.contiguous()
        else:
            self.error(f"Unexpected model output type: {type(output)}")
