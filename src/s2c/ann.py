import math
import torch
from torch import nn
import torch.nn.functional as F

def adjust_learning_rate(optimizer, factor=0.1, min_lr=0.00001, max_lr=0.0001, increase=False):
    """
    Adjusts the learning rate by a given factor and switches direction automatically 
    if it hits a min or max threshold.
    
    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer whose LR will be modified.
        factor (float): The factor by which to adjust the LR.
            - If increasing, LR = LR / factor
            - If decreasing, LR = LR * factor
        min_lr (float): Lower bound for the learning rate.
        max_lr (float): Upper bound for the learning rate.
        increase (bool): Whether to increase the LR. If False, LR decreases.
    
    Returns:
        bool: The new direction flag (True if we should now increase, False if decrease).
    """
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        
        # Apply the increase or decrease
        if increase:
            new_lr = old_lr / factor
        else:
            new_lr = old_lr * factor
        
        # Clip the new_lr to ensure it doesn't go below min_lr or above max_lr
        if new_lr < min_lr:
            new_lr = min_lr
            increase = True  # Next time, we want to increase since we hit the lower bound
        elif new_lr > max_lr:
            new_lr = max_lr
            increase = False  # Next time, we want to decrease since we hit the upper bound
        
        param_group['lr'] = new_lr
        print(f"Adjusted LR from {old_lr:.6f} to {new_lr:.6f} | Next step increase={increase}")

    return increase


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class NeutalNetA(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim,
        l1=512,
        l2=256,
        l3=128,
        dropout=0.3
    ):
        super().__init__()
        activation = nn.ReLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, l1))
        self.layers.append(activation)
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(l1, l2))
        self.layers.append(activation)
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(l2, l3))
        self.layers.append(activation)
        self.layers.append(nn.Linear(l3, output_dim))
        self.layers.append(nn.Softmax(dim=1))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralNetB(nn.Module):
    def __init__(
        self, 
        input_channels,
        sequence_length,
        output_dim,
        conv1_channels=32,
        conv2_channels=64,
        dense1=512,
        dense2=256,
        dense3=128,
        dropout=0.3
    ):
        super().__init__()
        activation = nn.ReLU()
        
        # Adjust kernel size based on sequence length
        kernel_size = min(3, sequence_length)
        padding = kernel_size // 2  # Same padding
        
        # Calculate the size after convolutions with padding
        conv1_length = sequence_length  # Same padding maintains length
        conv2_length = conv1_length     # Same padding maintains length
        flattened_size = conv2_channels * conv2_length
        
        self.layers = nn.ModuleList([
            nn.Conv1d(input_channels, conv1_channels, kernel_size=kernel_size, padding=padding),
            activation,
            nn.BatchNorm1d(conv1_channels),
            nn.Dropout(dropout),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, padding=padding),
            activation,
            nn.BatchNorm1d(conv2_channels),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(flattened_size, dense1),
            activation,
            nn.Linear(dense1, dense2),
            activation,
            nn.Linear(dense2, dense3),
            activation,
            nn.Linear(dense3, output_dim),
            nn.Softmax(dim=1)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralNetC(nn.Module):
    def __init__(
        self, 
        input_channels,
        sequence_length,
        output_dim,
        conv1_channels=64,
        conv2_channels=128,
        conv3_channels=256,
        dense1=512,
        dense2=256,
        dropout=0.3
    ):
        super().__init__()
        activation = nn.ReLU()
        
        # Adjust kernel size based on sequence length
        kernel_size = min(3, sequence_length)
        padding = kernel_size // 2  # Same padding
        
        # Calculate the size after convolutions with padding
        conv1_length = sequence_length  # Same padding maintains length
        conv2_length = conv1_length     # Same padding maintains length
        conv3_length = conv2_length     # Same padding maintains length
        flattened_size = conv3_channels * conv3_length
        
        self.layers = nn.ModuleList([
            nn.Conv1d(input_channels, conv1_channels, kernel_size=kernel_size, padding=padding),
            activation,
            nn.BatchNorm1d(conv1_channels),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel_size, padding=padding),
            activation,
            nn.BatchNorm1d(conv2_channels),
            nn.Conv1d(conv2_channels, conv3_channels, kernel_size=kernel_size, padding=padding),
            activation,
            nn.BatchNorm1d(conv3_channels),
            nn.Flatten(),
            nn.Linear(flattened_size, dense1),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dense1, dense2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dense2, output_dim),
            nn.Softmax(dim=1)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


## Bi-directional LSTM
class NeuralNetD(nn.Module):
    """Bidirectional LSTM-based neural network for sequence classification."""

    def __init__(
        self,
        input_channels,
        output_dim,
        projection_size=64,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dense1=256,
        dense2=128,
        dropout=0.3
    ):
        """
        Initialize the bidirectional LSTM model.

        Args:
            input_channels: Number of input features per time step
            output_dim: Number of output classes
            projection_size: Size of the input projection layer (0 to disable)
            lstm_hidden_size: Hidden size of LSTM layers
            lstm_num_layers: Number of LSTM layers
            dense1: Size of first dense layer
            dense2: Size of second dense layer
        """
        super().__init__()

        self.projection_size = projection_size

        if self.projection_size > 0:
            self.input_proj = nn.Linear(input_channels, projection_size)
        

        self.lstm = nn.LSTM(
            projection_size if self.projection_size > 0 else input_channels,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(2 * lstm_hidden_size, dense1)
        self.fc2 = nn.Linear(dense1, dense2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dense2, output_dim)

    def forward(self, x):
        """Forward pass through the model."""
        if self.projection_size > 0:
            x = self.input_proj(x)
            out, (h_n, c_n) = self.lstm(x)
        else:
            out, (h_n, c_n) = self.lstm(x)
        # Concatenate the final hidden states from both directions
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        feats = torch.cat((h_fwd, h_bwd), dim=1)

        feats = self.dropout(self.relu(self.fc1(feats)))
        feats = self.dropout(self.relu(self.fc2(feats)))
        out = self.out(feats)
        return out


## Bi-directional GRU
class NeuralNetE(nn.Module):
    """Bidirectional GRU-based neural network for sequence classification."""

    def __init__(
        self,
        input_channels,
        output_dim,
        projection_size=64,
        gru_hidden_size=128,
        gru_num_layers=2,
        dense1=256,
        dense2=128,
        dropout=0.3
    ):
        """
        Initialize the bidirectional GRU model.

        Args:
            input_channels: Number of input features per time step
            output_dim: Number of output classes
            projection_size: Size of the input projection layer (0 to disable)
            gru_hidden_size: Hidden size of GRU layers
            gru_num_layers: Number of GRU layers
            dense1: Size of first dense layer
            dense2: Size of second dense layer
        """
        super().__init__()

        self.projection_size = projection_size
        
        if self.projection_size > 0:
            self.input_proj = nn.Linear(input_channels, projection_size)
        

        self.gru = nn.GRU(
            projection_size if self.projection_size > 0 else input_channels,
            gru_hidden_size,
            gru_num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(2 * gru_hidden_size, dense1)
        self.fc2 = nn.Linear(dense1, dense2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dense2, output_dim)

    def forward(self, x):
        """Forward pass through the model."""
        if self.projection_size > 0:
            x = self.input_proj(x)
            out, h_n = self.gru(x)
        else:
            out, h_n = self.gru(x)
        # Concatenate the final hidden states from both directions
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        feats = torch.cat((h_fwd, h_bwd), dim=1)

        feats = self.dropout(self.relu(self.fc1(feats)))
        feats = self.dropout(self.relu(self.fc2(feats)))
        out = self.out(feats)
        return out


# Transformer-based model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d_model)
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d_model)
        N, T, D = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # split into heads: (N, h, T, d_k)
        q = q.view(N, T, self.h, self.d_k).transpose(1, 2)
        k = k.view(N, T, self.h, self.d_k).transpose(1, 2)
        v = v.view(N, T, self.h, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (N, h, T, T)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (N, h, T, d_k)

        # concat heads: (N, T, d_model)
        out = out.transpose(1, 2).contiguous().view(N, T, D)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual + norm
        x = self.norm1(x + self.drop(self.self_attn(x)))
        # FFN + residual + norm
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class NeuralNetG(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        seq_len: int = 69,
        num_classes: int = 16,
        d_model: int = 128,
        num_heads: int = 8,
        d_ff: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (N, 69, 4)
        returns logits: (N, 16)
        """
        x = self.input_proj(x)      # (N, T, d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)            # (N, T, d_model)

        pooled = x.mean(dim=1)      # (N, d_model)  mean pooling over time
        logits = self.classifier(pooled)  # (N, num_classes)
        return logits