import math
import torch
import torch.nn as nn
from typing import Optional


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding Module.
    
    Args:
        d_model (int): Dimension of the model embeddings.
        max_seq_length (int): Maximum sequence length the model will encounter.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize learnable positional encoding."""
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create learnable embeddings: shape (1, max_seq_length, d_model)
        # The 1 at the start is for batch broadcasting
        self.pe = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        # Initialize with small values
        nn.init.uniform_(self.pe, -0.02, 0.02)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            Tensor with positional encodings added, same shape as input.
        """
        seq_len = x.size(1)
        
        # Get positional encodings for current sequence length
        pe = self.pe[:, :seq_len, :]
        
        # Add to input (broadcasting handles batch dimension)
        x = x + pe
        
        return self.dropout(x)


class LSPE(nn.Module):
    """
    Learnable Sinusoidal Positional Encoding (LSPE).
    
    Hybrid approach combining sinusoidal positional encoding with learnable
    scaling and bias parameters. This provides structure from sinusoids while
    allowing the model to learn task-specific adjustments.
    
    Args:
        d_model (int): Dimension of the model embeddings.
        max_seq_length (int): Maximum sequence length the model will encounter.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    
    Example:
        >>> lspe = LSPE(d_model=512, max_seq_length=5000)
        >>> x = torch.randn(32, 100, 512)  # batch_size=32, seq_len=100
        >>> output = lspe(x)
        >>> output.shape
        torch.Size([32, 100, 512])
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int,
        dropout: float = 0.1
    ) -> None:
        """Initialize learnable sinusoidal positional encoding."""
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Pre-compute sinusoidal positional encoding patterns (non-trainable)
        pe_sin = self._compute_pe_sin(max_seq_length, d_model)
        pe_cos = self._compute_pe_cos(max_seq_length, d_model)
        
        # Register as buffers (not trainable, but move with model)
        self.register_buffer("pe_sin", pe_sin)
        self.register_buffer("pe_cos", pe_cos)
        
        # Learnable scaling factors for sine and cosine
        self.scale_sin = nn.Parameter(torch.ones(d_model // 2))
        self.scale_cos = nn.Parameter(torch.ones(d_model // 2))
        
        # Learnable bias term
        self.bias = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def _compute_pe_sin(max_seq_length: int, d_model: int) -> torch.Tensor:
        """Compute sinusoidal positional encoding matrix."""
        position = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        pe_sin = torch.sin(position * div_term)
        return pe_sin
    
    @staticmethod
    def _compute_pe_cos(max_seq_length: int, d_model: int) -> torch.Tensor:
        """Compute cosinusoidal positional encoding matrix."""
        position = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        pe_cos = torch.cos(position * div_term)
        return pe_cos
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable sinusoidal positional encodings to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            Tensor with positional encodings added, same shape as input.
        """
        seq_len = x.size(1)
        
        # Get sinusoidal patterns for current sequence length
        pe_sin = self.pe_sin[:seq_len, :]  # (seq_len, d_model // 2)
        pe_cos = self.pe_cos[:seq_len, :]  # (seq_len, d_model // 2)
        
        # Apply learnable scaling
        pe_sin_scaled = pe_sin * self.scale_sin
        pe_cos_scaled = pe_cos * self.scale_cos
        
        # Interleave sine and cosine components
        pe = torch.zeros(seq_len, self.d_model, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = pe_sin_scaled      # Even indices: sine
        pe[:, 1::2] = pe_cos_scaled      # Odd indices: cosine
        
        # Add bias
        pe = pe + self.bias
        
        # Reshape for broadcasting: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Add to input
        x = x + pe
        
        return self.dropout(x)
