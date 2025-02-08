"""
Defines custom loss functions for training deep learning models in PyTorch.  
Includes implementations for specialized loss functions tailored for classification tasks.  
Additional loss functions for other tasks (e.g., object detection, segmentation) may be added in the future.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def roc_curve_gpu(y_true, y_score):

    """
    Compute ROC curve on GPU. This function calculates the True Positive Rate (TPR)
    and False Positive Rate (FPR) for various thresholds.

    Arguments:
        y_true (Tensor): Ground truth labels, shape [batch_size].
        y_score (Tensor): Predicted scores/probabilities for each class, shape [batch_size] (binary) or [batch_size, num_classes] (multiclass).

    Returns:
        fpr (Tensor): False positive rate at each threshold.
        tpr (Tensor): True positive rate at each threshold.
    """

    # Handle binary or multiclass cases
    if y_score.dim() == 1:
        # Binary case: Treat y_score as probabilities for the positive class
        sorted_scores, sorted_indices = torch.sort(y_score, descending=True)
        num_classes = 2
    else:
        # Multiclass case
        num_classes = y_score.size(1)
        fpr_list, tpr_list = [], []
        for i in range(num_classes):
            class_probs = y_score[:, i]
            class_true = (y_true == i).float()
            sorted_scores, sorted_indices = torch.sort(class_probs, descending=True)

            # Compute TPR and FPR for this class
            tpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
            fpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
            total_positive = torch.sum(class_true).item()
            total_negative = class_true.size(0) - total_positive

            for j in range(sorted_scores.size(0)):
                threshold = sorted_scores[j]
                predictions = (class_probs >= threshold).float()
                tp = torch.sum((predictions == 1) & (class_true == 1)).item()
                fp = torch.sum((predictions == 1) & (class_true == 0)).item()
                tn = torch.sum((predictions == 0) & (class_true == 0)).item()
                fn = torch.sum((predictions == 0) & (class_true == 1)).item()

                tpr[j] = tp / (tp + fn) if tp + fn > 0 else 0
                fpr[j] = fp / (fp + tn) if fp + tn > 0 else 0

            fpr_list.append(fpr)
            tpr_list.append(tpr)

        return torch.stack(fpr_list), torch.stack(tpr_list)

    # For binary case
    tpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
    fpr = torch.zeros(sorted_scores.size(0), device=y_score.device)
    total_positive = torch.sum(y_true).item()
    total_negative = y_true.size(0) - total_positive

    for i in range(sorted_scores.size(0)):
        threshold = sorted_scores[i]
        predictions = (y_score >= threshold).float()
        tp = torch.sum((predictions == 1) & (y_true == 1)).item()
        fp = torch.sum((predictions == 1) & (y_true == 0)).item()
        tn = torch.sum((predictions == 0) & (y_true == 0)).item()
        fn = torch.sum((predictions == 0) & (y_true == 1)).item()

        tpr[i] = tp / (tp + fn) if tp + fn > 0 else 0
        fpr[i] = fp / (fp + tn) if fp + tn > 0 else 0

    return fpr, tpr

class CrossEntropyPAUCLoss(nn.Module):

    """
    Custom loss combining Cross-Entropy and Partial AUC (pAUC) optimization for multi-class classification.
    
    Arguments:
        recall_range (tuple): Range for recall (True Positive Rate) used in pAUC calculation (start, end).
        lambda_pauc (float): Weight for the pAUC calculation or contribution of pAUC in the loss function: 0.0 <= lambda_pauc <= 1.0. Default: 0.5
        num_classes (int): Number of classes for classification.
        label_smoothing (float): Factor for label smoothing to reduce overfitting.
        weight (Tensor): Optional class weights for handling class imbalance.
    """

    def __init__(
            self,
            recall_range=(0.95, 1.0),
            lambda_pauc=0.5,
            num_classes=2,
            label_smoothing=0.1,
            debug_mode=False,
            weight=None):
        
        super(CrossEntropyPAUCLoss, self).__init__()
        self.recall_range = recall_range
        self.max_pauc = recall_range[1] - recall_range[0]
        self.lambda_pauc = lambda_pauc
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.debug_mode = debug_mode

        # Register weight tensor (to handle class imbalance)
        if weight is not None:
            self.weight = weight.clone().detach().to(dtype=torch.float32)
        else:
            self.weight = torch.ones(num_classes, dtype=torch.float32)

    def forward(self, predictions, targets):

        """
        Computes the loss that combines Cross-Entropy and Partial AUC (pAUC) for multi-class classification.

        Arguments:
            predictions (Tensor): The model's raw output (logits), shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].

        Returns:
            Tensor: The computed total loss (Cross-Entropy + pAUC).
        """
        
        # Convert to probabilities for each class using softmax
        probs = F.softmax(predictions, dim=1)  # Shape: [batch_size, num_classes]

        # Convert targets to one-hot encoding
        targets = targets.to(predictions.device)
        targets_one_hot = torch.eye(self.num_classes, device=predictions.device)[targets]

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + self.label_smoothing / self.num_classes

        # Compute AUC for each class
        pauc=0
        p_auc_values = []
        if self.num_classes == 2:
            class_probs = probs[:, 1]
            class_targets = (targets_one_hot[:, 1] > 0.5).float()  # Threshold at 0.5
            
            # Compute ROC curve (assumed function available)
            fpr_vals, tpr_vals = roc_curve_gpu(class_targets, class_probs)

            # Compute the mask for the recall range
            recall_mask = (tpr_vals >= self.recall_range[0]) & (tpr_vals <= self.recall_range[1])
           
            if recall_mask.sum() > 0:
                # Compute weighted partial AUC using trapezoidal rule
                pauc = torch.trapz(torch.clamp(tpr_vals[recall_mask] - self.recall_range[0], min=0), fpr_vals[recall_mask])
                p_auc_values.append(pauc * self.weight[1].to(predictions.device))
            else:
                p_auc_values.append(torch.tensor(0.0, device=predictions.device))
        else:
            for i in range(self.num_classes):
                class_probs = probs[:, i]
                class_targets = (targets_one_hot[:, i] > 0.5).float()  # Threshold at 0.5
                
                # Compute ROC curve (assumed function available)
                fpr_vals, tpr_vals = roc_curve_gpu(class_targets, class_probs)

                # Compute the mask for the recall range
                recall_mask = (tpr_vals >= self.recall_range[0]) & (tpr_vals <= self.recall_range[1])

                if recall_mask.sum() > 0:
                    # Compute weighted partial AUC using trapezoidal rule
                    pauc = torch.trapz(torch.clamp(tpr_vals[recall_mask] - self.recall_range[0], min=0), fpr_vals[recall_mask])
                    p_auc_values.append(pauc * self.weight[i].to(predictions.device))
                else:
                    p_auc_values.append(torch.tensor(0.0, device=predictions.device))

        # Weighted mean of pAUC across all classes
        avg_p_auc = torch.sum(torch.stack(p_auc_values)) / (self.weight.sum().to(predictions.device) * self.max_pauc)
        avg_p_auc = torch.clamp(avg_p_auc, min=0.0, max=1.0)

        # Compute the weighted Cross-Entropy Loss with label smoothing
        log_probs = F.log_softmax(predictions, dim=1)
        ce_loss = -(targets_one_hot * log_probs * self.weight.to(predictions.device)).sum(dim=1).mean()

        # Compute pauc loss
        pauc_loss = 1 - torch.pow(avg_p_auc, 2.0) #-torch.log(avg_p_auc + 1e-7)

        # Compute total Loss
        total_loss = ((1 - self.lambda_pauc) * ce_loss) + (self.lambda_pauc * pauc_loss)

        # Debug mode
        if self.debug_mode:
           print(
               f"pauc: {torch.round(torch.tensor(pauc) * 10000) / 10000}, "
               f"pauc_loss: {torch.round(torch.tensor(pauc_loss) * 10000) / 10000}, "
               f"avg_p_auc: {torch.round(torch.tensor(avg_p_auc) * 10000) / 10000}, "
               f"ce_loss: {torch.round(torch.tensor(ce_loss) * 10000) / 10000}, "
               f"total_loss: {torch.round(torch.tensor(total_loss) * 10000) / 10000}"
               )

        return total_loss


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
            print(
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
        

# Define knowledge distillation loss
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
            alpha (float): The weight for the hard loss. Defaults to 0.5.
            temperature (float): The temperature for softening the logits. Defaults to 3.0.
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
