import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

# Define a classifier using Facebook's Wav2Vec2 models
class Wav2Vec2Classifier(nn.Module):
    
    """
    Wav2Vec2Classifier leverages pre-trained Facebook Wav2Vec2 models for audio classification.
    
    Available Pretrained Models:
    - facebook/wav2vec2-base
    - facebook/wav2vec2-large
    - facebook/wav2vec2-large-robust
    - facebook/wav2vec2-xlsr-53
    - facebook/wav2vec2-large-xlsr-53
    
    Args:
        base_model_name (str): The name of the pre-trained Wav2Vec2 model from Hugging Face.
        num_classes (int): Number of target classes for classification.

    Forward Pass:
        Input:  (batch_size, sequence_length) - Raw waveform audio signals.
        Output: (batch_size, num_classes) - Logits for each class.
    """

    def __init__(
        self,
        base_model_name="facebook/wav2vec2-base",
        num_classes=35
    ):
        super().__init__()

        # Load the pre-trained Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(base_model_name)

        # Classification head: Maps Wav2Vec2 embeddings to class logits
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_classes)

    def forward(self, x):
        
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Raw audio input tensor of shape (batch, sequence_length).
        
        Returns:
            torch.Tensor: Class logits of shape (batch, num_classes).
        """

        # Extract features (batch, seq_len, hidden_size)
        x = self.wav2vec2(x).last_hidden_state  

        # Pooling to get (batch, hidden_size)
        x = x.mean(dim=1)  

        # Map to (batch, num_classes)
        x = self.classifier(x)

        return x