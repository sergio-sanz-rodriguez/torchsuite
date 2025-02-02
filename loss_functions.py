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
        

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyPAUCLossV2(nn.Module):
    """
    Combines Cross-Entropy (or BCE for binary classification) with Partial AUC (pAUC) optimization.
    
    Arguments:
        recall_range (tuple): Recall range for pAUC computation.
        lambda_pauc (float): Weight for pAUC contribution (0.0 <= lambda_pauc <= 1.0).
        num_classes (int): Number of classification classes.
        label_smoothing (float): Label smoothing factor.
        weight (Tensor): Optional class weights for imbalance handling.
        debug (bool): Enables debug output.
    """

    def __init__(
            self,
            recall_range=(0.95, 1.0),
            lambda_pauc=0.5,
            num_classes=2,
            label_smoothing=0.1,
            weight=None,
            debug_mode=False):
        
        super().__init__()
        self.recall_range = recall_range
        self.lambda_pauc = lambda_pauc
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.debug_mode = debug_mode

        # Loss function selection
        if num_classes == 2:
            self.cross_entropy_loss = nn.BCEWithLogitsLoss(weight=weight)
        else:
            self.corss_entropy_loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

        # Register weight tensor for pAUC computation
        self.weight = weight.clone().detach().to(dtype=torch.float32) if weight is not None else torch.ones(num_classes, dtype=torch.float32)

    def forward(self, predictions, targets):
        """
        Computes combined loss: Cross-Entropy (or BCE) + pAUC.

        Arguments:
            predictions (Tensor): Model logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].

        Returns:
            Tensor: Total loss value.
        """
        
        # Compute standard classification loss
        if self.num_classes == 2:
            # Modify targets to one-hot encoded format
            #targets = torch.eye(2, device=targets.device)[targets.long()].unsqueeze(1)
            targets = targets.unsqueeze(1)
            ce_loss = self.cross_entropy_loss(predictions, targets)
            # Convert to probability
            probs = torch.sigmoid(predictions).squeeze(1)
        else:
            # Convert logits to probabilities
            ce_loss = self.cross_entropy_loss(predictions, targets)
            probs = F.softmax(predictions, dim=1)

        # Compute pAUC loss
        pauc_loss = self.compute_pauc_loss(probs, targets)

        # Compute total loss
        total_loss = (1 - self.lambda_pauc) * ce_loss + self.lambda_pauc * pauc_loss

        if self.debug_mode:
            print(f"ce_loss: {ce_loss.item()}, pauc_loss: {pauc_loss.item()}, total_loss: {total_loss.item()}")

        return total_loss

    def compute_pauc_loss(self, probs, targets):
        """
        Computes the pAUC loss.

        Arguments:
            probs (Tensor): Predicted probabilities.
            targets (Tensor): True labels.

        Returns:
            Tensor: pAUC loss.
        """
        
        pauc_values = []

        # Binary classification case
        if self.num_classes == 2:

            # Compute the ROC curve values
            class_probs = probs
            class_targets = targets.float()
            fpr_vals, tpr_vals = roc_curve_gpu(class_targets, class_probs)

            # Mast to filter recall values within the specified recall range
            recall_mask = (tpr_vals >= self.recall_range[0]) & (tpr_vals <= self.recall_range[1])

            # If there are valid points in the recall range
            if recall_mask.sum() > 0:
                
                # compute pAUC using the trapezoidal rule and clamp the recall to the lower bound of the range
                pauc = torch.trapz(torch.clamp(tpr_vals[recall_mask] - self.recall_range[0], min=0), fpr_vals[recall_mask])
                pauc_values.append(torch.tensor(pauc, device=probs.device))

            else:

                # No valid points, append 0
                pauc_values.append(torch.tensor(0.0, device=probs.device))
        
        # Multi-class classification case
        else:
            for i in range(self.num_classes):
                
                # Compute the ROC curve values (False Positive Rate and True Positive Rate)
                class_probs = probs[:, i]
                class_targets = (targets == i).float()
                fpr_vals, tpr_vals = roc_curve_gpu(class_targets, class_probs)

                # Mask to filter recall values within the specified recall range
                recall_mask = (tpr_vals >= self.recall_range[0]) & (tpr_vals <= self.recall_range[1])

                # If there are valid points in the recall range
                if recall_mask.sum() > 0:

                    # Compute pAUC using the trapezoidal rule, clamp the recall to the lower bound of the range
                    pauc = torch.trapz(torch.clamp(tpr_vals[recall_mask] - self.recall_range[0], min=0), fpr_vals[recall_mask])
                    pauc_values.append(torch.tensor(pauc * self.weight[i], device=probs.device))

                else:

                    # No valid points, append 0
                    pauc_values.append(torch.tensor(0.0, device=probs.device))

        # Compute the average pAUC over all classes, weighted by the class weights
        avg_pauc = torch.sum(torch.stack(pauc_values)) / (self.weight.sum().to(probs.device) * (self.recall_range[1] - self.recall_range[0]))

        # Ensure the average pAUC is between 0 and 1
        avg_pauc = torch.clamp(avg_pauc, min=0.0, max=1.0)

        # Return the pAUC loss (1 - pAUC squared)
        return 1 - avg_pauc**2



