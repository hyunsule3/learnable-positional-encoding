"""
Simple training comparison example for positional encoding methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import LearnablePositionalEncoding, LSPE
from src.utils import test_generalization, plot_comparison


def create_dummy_dataset(
    num_samples: int = 1000,
    seq_length: int = 50,
    d_model: int = 256,
    batch_size: int = 32
) -> DataLoader:
    """
    Create a dummy dataset for demonstration.
    
    Args:
        num_samples: Number of samples.
        seq_length: Sequence length.
        d_model: Model dimension.
        batch_size: Batch size.
    
    Returns:
        DataLoader with dummy data.
    """
    # Generate random inputs and targets
    X = torch.randn(num_samples, seq_length, d_model)
    y = torch.randn(num_samples, seq_length, 1)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


class SimpleModel(nn.Module):
    """
    Simple model combining positional encoding with a linear projection.
    
    Args:
        d_model: Model dimension.
        positional_encoding: Positional encoding module.
    """
    
    def __init__(
        self,
        d_model: int,
        positional_encoding: nn.Module
    ) -> None:
        super().__init__()
        self.positional_encoding = positional_encoding
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
        
        Returns:
            Output tensor of shape (batch_size, seq_length, 1).
        """
        # Add positional encodings
        x = self.positional_encoding(x)
        
        # Project to output
        x = self.linear(x)
        
        return x


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
    
    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model.
    
    Args:
        model: Model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.
    
    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = None
) -> dict:
    """
    Train a model with positional encoding.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        device: Device to train on.
    
    Returns:
        Dictionary with training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    return history


def compare_methods(
    d_model: int = 256,
    max_seq_length: int = 200,
    train_seq_length: int = 50,
    num_epochs: int = 10,
    batch_size: int = 32,
) -> dict:
    """
    Compare learnable positional encoding methods.
    
    Args:
        d_model: Model dimension.
        max_seq_length: Maximum sequence length.
        train_seq_length: Training sequence length.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
    
    Returns:
        Dictionary with comparison results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create data
    print("Creating dummy dataset...")
    train_loader = create_dummy_dataset(
        num_samples=500,
        seq_length=train_seq_length,
        d_model=d_model,
        batch_size=batch_size
    )
    val_loader = create_dummy_dataset(
        num_samples=100,
        seq_length=train_seq_length,
        d_model=d_model,
        batch_size=batch_size
    )
    
    results = {}
    
    # Method 1: LearnablePositionalEncoding
    print("\n" + "=" * 60)
    print("Method 1: LearnablePositionalEncoding")
    print("=" * 60)
    
    pos_enc_learnable = LearnablePositionalEncoding(
        d_model=d_model,
        max_seq_length=max_seq_length,
        dropout=0.1
    )
    model_learnable = SimpleModel(
        d_model=d_model,
        positional_encoding=pos_enc_learnable
    )
    
    history_learnable = train_model(
        model_learnable,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        device=device
    )
    
    # Test generalization
    gen_learnable = test_generalization(
        pos_enc_learnable,
        train_length=train_seq_length,
        test_lengths=[75, 100, 150],
        d_model=d_model
    )
    
    results["LearnablePositionalEncoding"] = {
        "history": history_learnable,
        "generalization": gen_learnable,
        "model": model_learnable
    }
    
    # Method 2: LSPE
    print("\n" + "=" * 60)
    print("Method 2: LSPE (Learnable Sinusoidal Positional Encoding)")
    print("=" * 60)
    
    lspe = LSPE(
        d_model=d_model,
        max_seq_length=max_seq_length,
        dropout=0.1
    )
    model_lspe = SimpleModel(
        d_model=d_model,
        positional_encoding=lspe
    )
    
    history_lspe = train_model(
        model_lspe,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        device=device
    )
    
    # Test generalization
    gen_lspe = test_generalization(
        lspe,
        train_length=train_seq_length,
        test_lengths=[75, 100, 150],
        d_model=d_model
    )
    
    results["LSPE"] = {
        "history": history_lspe,
        "generalization": gen_lspe,
        "model": model_lspe
    }
    
    return results


def print_summary(results: dict) -> None:
    """
    Print comparison summary.
    
    Args:
        results: Comparison results from compare_methods().
    """
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    for method_name, method_results in results.items():
        print(f"\n{method_name}:")
        print("-" * 40)
        
        # Final losses
        history = method_results["history"]
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        print(f"  Final Train Loss: {final_train_loss:.6f}")
        print(f"  Final Val Loss:   {final_val_loss:.6f}")
        
        # Model parameters
        num_params = sum(p.numel() for p in method_results["model"].parameters())
        print(f"  Total Parameters: {num_params:,}")
        
        # Generalization stats
        gen = method_results["generalization"]
        print(f"\n  Generalization to longer sequences:")
        for test_len, stats in gen["test_results"].items():
            print(f"    Length {test_len}:")
            print(f"      Mean shift: {stats['mean_diff']:.6f}")
            print(f"      Std shift:  {stats['std_diff']:.6f}")


if __name__ == "__main__":
    # Run comparison
    results = compare_methods(
        d_model=256,
        max_seq_length=200,
        train_seq_length=50,
        num_epochs=10,
        batch_size=32
    )
    
    # Print summary
    print_summary(results)
    
    print("\nTraining complete!")
