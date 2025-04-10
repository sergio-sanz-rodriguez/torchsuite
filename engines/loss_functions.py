"""
Defines custom loss functions for training deep learning models in PyTorch.  
Includes implementations for specialized loss functions tailored for classification tasks.  
Additional loss functions for other tasks (e.g., object detection, segmentation) may be added in the future.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import Logger

# Image/Audio Classification: Cross-Entropy + pAUC Loss
class CrossEntropyPAUCLoss(torch.nn.Module):

    """
    Custom loss combining Cross-Entropy and Partial AUC (pAUC) optimization for multi-class classification
    
    Args:
        recall_threshold (float): The recall threshold for pAUC calculation.
        lambda_pauc (float): Weight for pAUC loss in the total loss.
        num_classes (int): The number of classes for classification.
        label_smoothing (float): Smoothing factor for cross-entropy loss.
        weight (tensor or None): Class weights for balancing the loss.
    """

    def __init__(
            self,
            recall_threshold=0.0,
            lambda_pauc=0.5,
            num_classes=2,
            label_smoothing=0.1,
            weight=None):
        
        """
        Initializes the loss function.
        
        Args:
            recall_threshold (float): The recall threshold for pAUC calculation.
            lambda_pauc (float): Weight for pAUC loss in the total loss.
            num_classes (int): The number of classes for classification.
            label_smoothing (float): Smoothing factor for cross-entropy loss.
            weight (tensor or None): Class weights for balancing the loss.
        """
        
        super().__init__()
        self.recall_threshold = recall_threshold        
        self.lambda_pauc = lambda_pauc
        self.num_classes = num_classes
        
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.tensor(weight, dtype=torch.float32)

        self.loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=weight
            )
    
    @staticmethod
    def roc_curve_gpu(y_score: torch.Tensor, y_true: torch.Tensor):

        """
        Vectorized ROC computation on GPU for binary classification.

        Args:
            y_score (Tensor): Predicted scores.
            y_true (Tensor): True binary labels.

        Returns:
            fpr (Tensor): False Positive Rate values.
            tpr (Tensor): True Positive Rate values.
        """

        device = y_score.device

        # Sort scores and corresponding true labels
        desc_score_indices = torch.argsort(y_score, descending=True)
        y_true_sorted = y_true[desc_score_indices]
        y_score_sorted = y_score[desc_score_indices]

        # True positives and false positives
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # Total positives and negatives
        total_positives = tps[-1].clamp(min=1.0)  # avoid divide by zero
        total_negatives = fps[-1].clamp(min=1.0)

        # Compute TPR and FPR
        tpr = tps / total_positives
        fpr = fps / total_negatives

        return fpr, tpr

    @staticmethod
    def trapezoidal_auc(fpr: torch.Tensor, tpr: torch.Tensor):

        """
        Calculate the AUC using the trapezoidal rule.

        Args:
            fpr (Tensor): False Positive Rate values.
            tpr (Tensor): True Positive Rate values.

        Returns:
            auc (Tensor): The area under the ROC curve.
        """

        # Assumes sorted fpr
        delta_fpr = fpr[1:] - fpr[:-1]
        avg_tpr = (tpr[1:] + tpr[:-1]) / 2
        return torch.sum(delta_fpr * avg_tpr)

    def calculate_pauc_at_recall(self, y_pred, y_true, macro=True):

        """
        Calculates the Partial AUC (pAUC) at a given recall threshold.
        This is vectorized for speed and GPU compatibility.

        Args:
            y_pred (Tensor): The predicted probabilities.
            y_true (Tensor): The true labels.
            macro (bool): If True, calculates macro pAUC (average across classes).

        Returns:
            pauc (Tensor): The calculated partial AUC.
        """

        partial_auc_values = []

        for class_idx in range(self.num_classes):
            y_scores_class = y_pred[:, class_idx]
            y_true_bin = (y_true == class_idx).float()

            if macro and torch.sum(y_true_bin) == 0:
                continue

            fpr, tpr = self.roc_curve_gpu(y_scores_class, y_true_bin)

            max_fpr = 1.0 - self.recall_threshold
            mask = fpr <= max_fpr

            if torch.any(mask):
                # Interpolate at max_fpr if needed
                idx = torch.where(~mask)[0]
                if idx.numel() > 0:
                    first_idx = idx[0]
                    prev_idx = first_idx - 1

                    # Linearly interpolate between prev and first_idx
                    fpr_prev, fpr_next = fpr[prev_idx], fpr[first_idx]
                    tpr_prev, tpr_next = tpr[prev_idx], tpr[first_idx]

                    slope = (tpr_next - tpr_prev) / (fpr_next - fpr_prev + 1e-8)
                    tpr_interp = tpr_prev + slope * (max_fpr - fpr_prev)

                    fpr = torch.cat([fpr[mask], max_fpr.unsqueeze(0)])
                    tpr = torch.cat([tpr[mask], tpr_interp.unsqueeze(0)])
                else:
                    # All points are within max_fpr
                    pass
            else:
                # All fpr > max_fpr, use default line
                fpr = torch.tensor([0.0, max_fpr], device=y_pred.device)
                tpr = torch.tensor([0.0, 0.0], device=y_pred.device)

            partial_auc = self.trapezoidal_auc(fpr, tpr)
            partial_auc_values.append(partial_auc)

        return torch.mean(torch.stack(partial_auc_values))

    def forward(self, predictions, targets):

        """
        Forward pass for the loss function, combining cross-entropy and pAUC.

        Args:
            predictions (Tensor): Raw predictions (logits) from the model.
            targets (Tensor): Ground truth labels.

        Returns:
            total_loss (Tensor): The combined loss value (Cross-Entropy + pAUC).
        """
        
        # Cross-Entropy uses raw logits, pAUC needs probabilities
        probs = torch.nn.functional.softmax(predictions, dim=1)

        # Compute Cross-Entropy Loss
        ce_loss = self.loss_fn(predictions, targets)

        # Compute pAUC Loss
        pauc = self.calculate_pauc_at_recall(probs, targets)
        pauc_loss = 1 - torch.pow(torch.tensor(pauc, device=predictions.device), 2.0)

        # Total Loss
        total_loss = ((1 - self.lambda_pauc) * ce_loss) + (self.lambda_pauc * pauc_loss)

        return total_loss



# Image/Audio classification: Cross-Entropy + FPR Loss
class CrossEntropyFPRLoss(nn.Module):

    """
    Custom loss combining Cross-Entropy and False Positive Rate (FPR) optimization for binary and multi-class classification.
    
    Arguments:
        alpha (float): Scaling factor for the FPR loss. Should be between 0.1 and 1.0.
        recall_threshold (float): The recall value at which FPR is calculated (default is 0.95).
        num_classes (int): Number of classes for classification.
        label_smoothing (float): Factor for label smoothing to reduce overfitting.
        weight (Tensor): Optional class weights for handling class imbalance.
        debug (bool): If True, prints debug information.
    """

    def __init__(self,
                 alpha=0.5,
                 recall_threshold=0.95,
                 num_classes=2,
                 label_smoothing=0.1,
                 weight=None,
                 debug_mode=False):
        super(CrossEntropyFPRLoss, self).__init__()
        
        # Initialize class parameters
        self.alpha = torch.clamp(alpha, min=0.1, max=1.0)
        self.recall_threshold = recall_threshold
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.debug_mode = debug_mode
        
        # Handle class weights (to deal with class imbalance)
        if weight is not None:
            self.weight = weight.clone().detach().to(dtype=torch.float32)
        else:
            self.weight = torch.ones(num_classes, dtype=torch.float32)

    def forward(self, predictions, targets):

        """
        Computes the loss that combines Cross-Entropy and False Positive Rate (FPR) for both binary and multi-class classification.

        Arguments:
            predictions (Tensor): The model's raw output (logits), shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].

        Returns:
            Tensor: The computed total loss (Cross-Entropy + FPR).
        """
        
        # Convert to probabilities for each class using softmax
        probs = F.softmax(predictions, dim=1)  # Shape: [batch_size, num_classes]

        # Convert targets to one-hot encoding
        targets = targets.to(predictions.device)
        targets_one_hot = torch.eye(self.num_classes, device=predictions.device)[targets]

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + self.label_smoothing / self.num_classes

        # Compute the weighted Cross-Entropy Loss with label smoothing
        log_probs = F.log_softmax(predictions, dim=1)
        ce_loss = -(targets_one_hot * log_probs * self.weight.to(predictions.device)).sum(dim=1).mean()

        # Compute False Positive Rate (FPR) at the given recall threshold
        fpr_values = []
        if self.num_classes == 2:
            class_probs = probs[:, 1]
            class_targets = (targets_one_hot[:, 1] > 0.5).float()  # Threshold at 0.5

            # Compute FPR for binary classification
            fpr = self.compute_fpr(class_targets, class_probs, self.recall_threshold)
            fpr_values.append(fpr)
        else:
            for i in range(self.num_classes):
                class_probs = probs[:, i]
                class_targets = (targets_one_hot[:, i] > 0.5).float()  # Threshold at 0.5

                # Compute FPR for multi-class classification
                fpr = self.compute_fpr(class_targets, class_probs, self.recall_threshold)
                fpr_values.append(fpr)

        # Average FPR across all classes
        avg_fpr = torch.mean(torch.tensor(fpr_values))

        # Compute FPR loss
        fpr_loss = torch.exp(self.alpha * avg_fpr) - 1  # Exponential FPR loss

        # Compute total Loss (weighted combination of Cross-Entropy and FPR loss)
        total_loss = ce_loss + fpr_loss

        # Debug mode (optional)
        if self.debug_mode:
            Logger().info(
                f"ce_loss: {torch.round(torch.tensor(ce_loss) * 10000) / 10000}, "
                f"fpr_loss: {torch.round(torch.tensor(fpr_loss) * 10000) / 10000}, "
                f"total_loss: {torch.round(torch.tensor(total_loss) * 10000) / 10000}"
            )

        return total_loss

    def compute_fpr(self, targets, probs, recall_threshold):

        """
        Compute False Positive Rate (FPR) at the given recall threshold.

        Arguments:
            targets (Tensor): Ground truth labels, shape [batch_size].
            probs (Tensor): Predicted probabilities, shape [batch_size].
            recall_threshold (float): The recall threshold for FPR calculation.
            
        Returns:
            float: The computed FPR.
        """
        # Sort probabilities and calculate True Positives (TP) and False Positives (FP)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_targets = targets[sorted_indices]

        # Compute True Positives and False Positives
        tp = torch.sum(sorted_targets)
        fp = torch.sum(1 - sorted_targets)

        # Compute the recall value
        recall = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Calculate FPR based on recall threshold
        if recall >= recall_threshold:
            fpr = fp / (fp + tp) if (fp + tp) > 0 else 0
            return fpr
        else:
            return 0
        

# Image distillation (classification) Loss
class DistillationLoss(nn.Module):

    """
    This class implements the knowledge distillation loss, which combines both
    a soft target loss (based on the teacher's logits) and a hard target loss 
    (based on the ground truth labels).

    The loss is a weighted sum of:
        - Soft loss: The Kullback-Leibler divergence between the teacher's and 
          student's softened logits.
        - Hard loss: The Cross Entropy loss between the student's logits and 
          the true labels.

    Attributes:
        alpha (float): Weight for the hard loss (CrossEntropyLoss). Defaults to 0.5.
        temperature (float): Temperature for softening the logits. Defaults to 3.0.
        kl_div (nn.KLDivLoss): The Kullback-Leibler divergence loss function.
        ce_loss (nn.CrossEntropyLoss): The Cross Entropy loss function.

    Methods:
        forward(student_logits, teacher_logits, labels):
            Computes the combined distillation loss.
    """

    def __init__(self, alpha=0.5, temperature=3.0, label_smoothing=0.1):

        """
        Initializes the DistillationLoss object with the specified alpha and temperature values.

        Args:
            alpha (float): Controls the weighting between soft and hard losses. Defaults to 0.5.
            temperature (float): Smooths the teacher’s probability distribution, making it easier for the student to learn from. Defaults to 3.0.
            label_smoothing: Controlls the confidence of the ground truth labels, helping to prevent overfitting. Defaults to 0.1.
        """

        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, student_logits, teacher_logits, labels):

        """
        Computes the combined distillation loss by calculating the soft loss and hard loss
        and returning their weighted sum.

        Args:
            student_logits (torch.Tensor): The logits produced by the student model.
            teacher_logits (torch.Tensor): The logits produced by the teacher model.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed distillation loss.
        """
        
        soft_loss = self.kl_div(
            torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        )
        hard_loss = self.ce_loss(student_logits, labels)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# Image segmentation: Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, num_classes=1, from_logits=True, eps=1e-6, reduction='mean'):
        
        """
        Dice Loss for binary and multi-class segmentation (flattened version for global calculation).
        
        Args:
            num_classes (int): Number of classes (1 for binary segmentation, >1 for multi-class).
            from_logits (bool): If True, applies sigmoid activation to predictions.
            eps (float): Small constant for numerical stability.
            reduction (str): Specifies the reduction method ('mean', 'sum', or None).
        """
        
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_true, y_pred):

        """
        Computes the Dice Loss by flattening both predictions and ground truth.

        Args:
            y_true: Ground truth labels (one-hot encoded for multi-class) -> shape (N, C, H, W)
            y_pred: Model predictions (logits if from_logits=True) -> shape (N, C, H, W)

        Returns:
            Dice Loss value (lower is better).
        """

        # Apply softmax / sigmoid
        if self.from_logits:
            y_pred = torch.softmax(y_pred, dim=1) if self.num_classes > 1 else torch.sigmoid(y_pred)

        # Ensure ground truth is float for stability
        y_true = y_true.float() 

        # Binary case (num_classes == 1)
        if self.num_classes == 1:

            # Compute intersection and union
            intersection = torch.sum(y_true * y_pred, dim=(2, 3))
            union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3))

        # Multi-class case (num_classes > 1)
        else:
            # Convert predictions to one-hot-like representation
            y_pred_one_hot = F.one_hot(torch.argmax(y_pred, dim=1), num_classes=self.num_classes)
            y_pred_one_hot = y_pred_one_hot.permute(0, 3, 1, 2).float()  # Shape (B, C, H, W)

            # Compute intersection and union
            intersection = torch.sum(y_true * y_pred_one_hot, dim=(2, 3))
            union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred_one_hot, dim=(2, 3))        
        
        # Compute Dice coefficient
        dice = (2. * intersection + self.eps) / (union + self.eps)
        if torch.any(dice < 0) or torch.any(dice > 1):
            Logger().warning(f"Dice loss: Dice coefficient out of range for class.")
        dice = torch.clamp(dice, 0.0, 1.0)
        dice_loss = 1 - dice

        # Reduce across classes
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss



# Image Segmentation: Dice + Cross-Entropy Loss
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=1, alpha=0.5, threshold=0.5, eps=1e-6, label_smoothing=0.1):
        
        """
        Combines Dice Loss and Cross-Entropy Loss.
        
        - alpha: weight factor (default: 0.5 means equal weight)
        """

        super(DiceCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.threshold = threshold
        self.eps = eps
        self.label_smoothing = label_smoothing        

        # For multi-class: CrossEntropyLoss, for binary classification: BCEWithLogitsLoss
        if self.num_classes > 1:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_true, y_pred):
        dice_loss = DiceLoss(self.num_classes, self.threshold, self.eps)(y_true, y_pred)
        ce_loss = self.ce_loss(y_pred, y_true.float())
        
        return self.alpha * dice_loss + (1 - self.alpha) * ce_loss
