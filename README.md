# Learnable Positional Encoding

A PyTorch implementation of learnable positional encoding methods for Transformer models.

## Overview

Positional encodings are crucial for Transformer models, conveying sequence position information to attention mechanisms. This repository implements and compares two approaches:

### 1. **Standard Learnable Positional Encoding**
Fully trainable embeddings where each position and dimension has a learnable parameter.

**Pros:**
- Maximum flexibility
- Learns arbitrary patterns

**Cons:**
- High parameter count: O(sequence_length × d_model)
- Poor generalization to longer sequences

### 2. **LSPE (Learnable Sinusoidal Positional Encoding)** ⭐ Recommended
Hybrid approach combining sinusoidal base patterns with learnable scaling and bias.

**Pros:**
- Parameter efficient: O(d_model)
- Excellent generalization to longer sequences
- Built-in structural inductive bias

**Cons:**
- Less flexible than fully learnable approach

## Project Structure

```
learnable-positional-encoding/
├── src/
│   ├── __init__.py
│   ├── positional_encoding.py    # Core classes
│   └── utils.py                  # Visualization & analysis utilities
├── examples/
│   └── train_comparison.py       # Training comparison script
├── results/plots/                # Output directory
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+

### Local Setup
```bash
# Clone the repository
git clone https://github.com/hyunsule3/learnable-positional-encoding.git
cd learnable-positional-encoding

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup
```python
# In a Colab cell:
!pip install torch numpy matplotlib

# Clone the repo (optional)
!git clone https://github.com/hyunsule3/learnable-positional-encoding.git
%cd learnable-positional-encoding
```

## Quick Start

### Basic Usage

```python
import torch
from src import LearnablePositionalEncoding, LSPE

# Create positional encoding
d_model = 512
max_seq_length = 5000

# Option 1: Learnable PE
pos_enc = LearnablePositionalEncoding(
    d_model=d_model,
    max_seq_length=max_seq_length,
    dropout=0.1
)

# Option 2: LSPE (Recommended)
lspe = LSPE(
    d_model=d_model,
    max_seq_length=max_seq_length,
    dropout=0.1
)

# Apply to embeddings
x = torch.randn(batch_size=32, seq_length=100, d_model=512)
encoded = lspe(x)  # Shape: (32, 100, 512)
```

### Run Training Comparison

```bash
python examples/train_comparison.py
```

This will:
- Create dummy data
- Train both methods
- Compare performance
- Test generalization

## API Reference

### LearnablePositionalEncoding

```python
class LearnablePositionalEncoding(nn.Module):
    """
    Standard learnable positional encoding with fully trainable embeddings.
    
    Args:
        d_model (int): Dimension of the model embeddings.
        max_seq_length (int): Maximum sequence length.
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        
        Returns:
            Tensor with positional encodings added
        """
```

### LSPE

```python
class LSPE(nn.Module):
    """
    Learnable Sinusoidal Positional Encoding (LSPE).
    
    Args:
        d_model (int): Dimension of the model embeddings.
        max_seq_length (int): Maximum sequence length.
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable sinusoidal positional encodings to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        
        Returns:
            Tensor with positional encodings added
        """
```

## Utility Functions

### Visualization
- `visualize_positional_encoding()` - Show encoding patterns as heatmap
- `visualize_similarity_matrix()` - Show cosine similarity between positions
- `compare_encodings()` - Compare multiple methods side-by-side

### Analysis
- `compute_similarity_matrix()` - Calculate cosine similarity
- `compute_orthogonality()` - Measure position distinctness
- `test_generalization()` - Test on longer sequences
- `get_encoding_stats()` - Get encoding statistics
- `plot_comparison()` - Plot generalization results

## Usage Examples

### Example 1: Compare Methods

```python
import torch
from src import LearnablePositionalEncoding, LSPE
from src.utils import compare_encodings, visualize_similarity_matrix

# Create both methods
lpe = LearnablePositionalEncoding(d_model=256, max_seq_length=200)
lspe = LSPE(d_model=256, max_seq_length=200)

# Compare side-by-side
fig = compare_encodings(
    {"Learnable PE": lpe, "LSPE": lspe},
    seq_length=100,
    d_model=256
)
fig.savefig("comparison.png")
```

### Example 2: Analyze Generalization

```python
from src.utils import test_generalization, plot_comparison

# Test how well LSPE generalizes to longer sequences
results_lspe = test_generalization(
    lspe,
    train_length=100,
    test_lengths=[150, 200, 300],
    d_model=256
)

# Plot results
fig = plot_comparison({"LSPE": results_lspe})
fig.savefig("generalization.png")
```

### Example 3: Get Statistics

```python
from src.utils import get_encoding_stats, compute_orthogonality

# Get encoding statistics
stats = get_encoding_stats(lspe, seq_length=100, d_model=256)
print(f"Mean: {stats['mean']:.4f}")
print(f"Std: {stats['std']:.4f}")

# Compute orthogonality
orthog = compute_orthogonality(lspe, seq_length=100, d_model=256)
print(f"Orthogonality: {orthog:.4f}")
```

## When to Use Each Method

| Scenario | Learnable PE | LSPE |
|----------|-------------|------|
| Fixed sequence length only | ✅ | ✅ |
| Variable length sequences | ❌ | ✅ |
| Limited parameter budget | ❌ | ✅ |
| Extrapolation needed | ❌ | ✅ |
| Maximum flexibility | ✅ | ⚠️ |

**Recommendation**: Use **LSPE** for most real-world applications.

## Performance Comparison

### Parameter Efficiency Example
For `d_model=256, max_seq_length=200`:
- **Learnable PE**: 51,200 parameters
- **LSPE**: 1,280 parameters
- **LSPE is 40x more efficient**

### Generalization
- **Learnable PE**: Cannot extrapolate beyond training length
- **LSPE**: Stable generalization to 3x+ longer sequences

## Running on Different Platforms

### Local Machine
```bash
# Install dependencies
pip install -r requirements.txt

# Run training example
python examples/train_comparison.py

# Run tests
python -c "from src import LearnablePositionalEncoding, LSPE; print('Success!')"
```

### Google Colab
```python
# Install dependencies
!pip install torch numpy matplotlib

# Clone repo
!git clone https://github.com/yourusername/learnable-positional-encoding.git
%cd learnable-positional-encoding

# Now use as normal
from src import LSPE
import torch

lspe = LSPE(d_model=256, max_seq_length=200)
x = torch.randn(32, 100, 256)
output = lspe(x)
print(output.shape)  # torch.Size([32, 100, 256])
```

## Testing Your Installation

```python
import torch
from src import LearnablePositionalEncoding, LSPE

# Create instances
lpe = LearnablePositionalEncoding(d_model=128, max_seq_length=100)
lspe = LSPE(d_model=128, max_seq_length=100)

# Test forward pass
x = torch.randn(2, 50, 128)
out_lpe = lpe(x)
out_lspe = lspe(x)

print(f"LearnablePositionalEncoding output shape: {out_lpe.shape}")
print(f"LSPE output shape: {out_lspe.shape}")

# Check parameters
print(f"LearnablePositionalEncoding params: {sum(p.numel() for p in lpe.parameters()):,}")
print(f"LSPE params: {sum(p.numel() for p in lspe.parameters()):,}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## References

- Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS.
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). "Self-Attention with Relative Position Representations."

## Frequently Asked Questions

**Q: Which method should I use?**  
A: Use LSPE unless you have specific reasons otherwise. It's more efficient and generalizes better.

**Q: Can I use these in my Transformer?**  
A: Yes! Just replace your positional encoding layer with either class.

**Q: Does this work on GPU?**  
A: Yes! Both methods work with `.to(device)` like any PyTorch module.

**Q: How do I handle sequences longer than max_seq_length?**  
A: Increase `max_seq_length` when creating the encoding. LSPE handles extrapolation well.

**Q: Can I fine-tune these?**  
A: Yes! Both are `nn.Module` subclasses with trainable parameters. Use standard PyTorch training.

---

**Last Updated**: January 2024
