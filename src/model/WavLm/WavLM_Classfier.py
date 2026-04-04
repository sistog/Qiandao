import torch
import torch.nn as nn
from transformers import WavLMModel


class WavLMClassifier(nn.Module):
    """WavLM-based classifier with options to freeze backbone and choose pooling.

    Args:
        num_classes (int): number of target classes
        freeze_backbone (bool): if True, freeze pretrained WavLM parameters
        pool_mode (str): one of 'mean', 'max', 'attention'
        attention_hidden (int): hidden dim for attention pooling (if used)
        dropout (float): optional dropout before classifier
    """

    def __init__(self, num_classes, freeze_backbone: bool = False, pool_mode: str = 'mean', attention_hidden: int = 128, dropout: float = 0.0):
        super(WavLMClassifier, self).__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.pool_mode = pool_mode
        hidden_size = self.wavlm.config.hidden_size

        # attention pooling module
        if self.pool_mode == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_size, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, 1)
            )
        else:
            self.attention_pool = None

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.classifier = nn.Linear(hidden_size, num_classes)

        if freeze_backbone:
            for param in self.wavlm.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.squeeze(1)
        outputs = self.wavlm(x)
        hidden_states = outputs.last_hidden_state  # [B, T, D]

        if self.pool_mode == 'mean':
            pooled_output = hidden_states.mean(dim=1)
        elif self.pool_mode == 'max':
            pooled_output, _ = hidden_states.max(dim=1)
        elif self.pool_mode == 'attention' and self.attention_pool is not None:
            # attention_pool produces score per frame -> softmax -> weighted sum
            scores = self.attention_pool(hidden_states)  # [B, T, 1]
            weights = torch.softmax(scores.squeeze(-1), dim=1).unsqueeze(-1)  # [B, T, 1]
            pooled_output = (hidden_states * weights).sum(dim=1)
        else:
            # fallback to mean
            pooled_output = hidden_states.mean(dim=1)

        if self.dropout is not None:
            pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return logits