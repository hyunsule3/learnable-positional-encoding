"""
Utility functions for positional encoding visualization and analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any


def compute_similarity_matrix(
    pos_encoding: nn.Module,
    seq_length: int = 100,
    d_model: int = 512
) -> torch.Tensor:
    """
    Compute cosine similarity between positional encodings at different positions.
    
    Args:
        pos_encoding: The positional encoding module.
        seq_length: Sequence length to analyze.
        d_model: Model dimension.
    
    Returns:
        Similarity matrix of shape (seq_length, seq_length).
    """
    # Create dummy input with zeros
    dummy_input = torch.zeros(1, seq_length, d_model)
    
    with torch.no_grad():
        # Get the positional encoding output
        output = pos_encoding(dummy_input)  # (1, seq_length, d_model)
        
        # Extract positional encodings by subtracting the input
        pe = output - dummy_input  # (1, seq_length, d_model)
        pe = pe.squeeze(0)  # (seq_length, d_model)
    
    # Normalize for cosine similarity
    pe_normalized = torch.nn.functional.normalize(pe, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(pe_normalized, pe_normalized.t())
    
    return similarity


def compute_orthogonality(
    pos_encoding: nn.Module,
    seq_length: int = 100,
    d_model: int = 512
) -> float:
    """
    Compute orthogonality of positional encodings.
    
    Higher values (closer to 1) indicate better orthogonality (more distinct positions).
    
    Args:
        pos_encoding: The positional encoding module.
        seq_length: Sequence length to analyze.
        d_model: Model dimension.
    
    Returns:
        Orthogonality score between 0 and 1.
    """
    similarity = compute_similarity_matrix(pos_encoding, seq_length, d_model)
    
    # Extract off-diagonal elements
    mask = torch.eye(seq_length, dtype=torch.bool)
    off_diagonal = similarity[~mask]
    
    # Average absolute off-diagonal similarity
    avg_similarity = torch.abs(off_diagonal).mean().item()
    
    # Orthogonality: 1 - avg_similarity (lower off-diagonal = more orthogonal)
    orthogonality = 1.0 - avg_similarity
    
    return orthogonality


def get_encoding_stats(
    pos_encoding: nn.Module,
    seq_length: int = 100,
    d_model: int = 512
) -> Dict[str, Any]:
    """
    Get statistics about the positional encodings.
    
    Args:
        pos_encoding: The positional encoding module.
        seq_length: Sequence length to analyze.
        d_model: Model dimension.
    
    Returns:
        Dictionary with statistics.
    """
    # Create dummy input
    dummy_input = torch.zeros(1, seq_length, d_model)
    
    with torch.no_grad():
        # Get the positional encoding output
        output = pos_encoding(dummy_input)  # (1, seq_length, d_model)
        # Extract positional encodings by subtracting the input
        pe = output - dummy_input  # (1, seq_length, d_model)
        pe = pe.squeeze(0)  # (seq_length, d_model)
    
    # Compute statistics
    stats = {
        "mean": pe.mean().item(),
        "std": pe.std().item(),
        "min": pe.min().item(),
        "max": pe.max().item(),
        "norm_mean": torch.norm(pe, dim=1).mean().item(),
        "norm_std": torch.norm(pe, dim=1).std().item(),
    }
    
    return stats


def test_generalization(
    pos_encoding: nn.Module,
    train_length: int = 100,
    test_lengths: Optional[list] = None,
    d_model: int = 512
) -> Dict[str, Any]:
    """
    Test how well encoding generalizes to longer sequences.
    
    Args:
        pos_encoding: The positional encoding module.
        train_length: Training sequence length.
        test_lengths: Sequence lengths to test. Defaults to [150, 200, 300].
        d_model: Model dimension.
    
    Returns:
        Dictionary with generalization results.
    """
    if test_lengths is None:
        test_lengths = [150, 200, 300]
    
    results = {
        "train_length": train_length,
        "test_results": {}
    }
    
    # Get training encodings
    dummy_train = torch.zeros(1, train_length, d_model)
    with torch.no_grad():
        output_train = pos_encoding(dummy_train)  # (1, train_length, d_model)
        pe_train = output_train - dummy_train  # (1, train_length, d_model)
        pe_train = pe_train.squeeze(0)  # (train_length, d_model)
    
    train_mean = pe_train.mean(dim=0)
    train_std = pe_train.std(dim=0)
    
    # Test on longer sequences
    for test_len in test_lengths:
        dummy_test = torch.zeros(1, test_len, d_model)
        with torch.no_grad():
            output_test = pos_encoding(dummy_test)  # (1, test_len, d_model)
            pe_test = output_test - dummy_test  # (1, test_len, d_model)
            pe_test = pe_test.squeeze(0)  # (test_len, d_model)
        
        test_mean = pe_test.mean(dim=0)
        test_std = pe_test.std(dim=0)
        
        # Measure difference from training distribution
        mean_diff = torch.abs(test_mean - train_mean).mean().item()
        std_diff = torch.abs(test_std - train_std).mean().item()
        
        results["test_results"][test_len] = {
            "mean_diff": mean_diff,
            "std_diff": std_diff,
        }
    
    return results


def visualize_positional_encoding(
    pos_encoding: nn.Module,
    seq_length: int = 100,
    d_model: int = 512,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize positional encoding as a heatmap.
    
    Args:
        pos_encoding: The positional encoding module.
        seq_length: Sequence length to visualize.
        d_model: Model dimension.
        figsize: Figure size.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    # Create dummy input
    dummy_input = torch.zeros(1, seq_length, d_model)
    
    with torch.no_grad():
        # Get the positional encoding output
        output = pos_encoding(dummy_input)  # (1, seq_length, d_model)
        # Extract positional encodings by subtracting the input
        pe = output - dummy_input  # (1, seq_length, d_model)
        pe = pe.squeeze(0).cpu().numpy()  # (seq_length, d_model)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Heatmap
    im1 = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Dimension")
    axes[0].set_title("Positional Encoding Heatmap")
    plt.colorbar(im1, ax=axes[0])
    
    # L2 norm over dimensions
    norms = np.linalg.norm(pe, axis=1)
    axes[1].plot(norms, linewidth=2)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("L2 Norm")
    axes[1].set_title("Encoding Norm per Position")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def visualize_similarity_matrix(
    pos_encoding: nn.Module,
    seq_length: int = 100,
    d_model: int = 512,
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize cosine similarity matrix of positional encodings.
    
    Args:
        pos_encoding: The positional encoding module.
        seq_length: Sequence length to analyze.
        d_model: Model dimension.
        figsize: Figure size.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    similarity = compute_similarity_matrix(pos_encoding, seq_length, d_model)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(similarity.cpu().numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    ax.set_title("Cosine Similarity Between Positional Encodings")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def compare_encodings(
    encodings_dict: Dict[str, nn.Module],
    seq_length: int = 100,
    d_model: int = 512,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple positional encoding methods side-by-side.
    
    Args:
        encodings_dict: Dictionary mapping names to encoding modules.
        seq_length: Sequence length to visualize.
        d_model: Model dimension.
        figsize: Figure size. Auto-calculated if None.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    n_methods = len(encodings_dict)
    
    if figsize is None:
        figsize = (5 * n_methods, 5)
    
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (name, encoding) in enumerate(encodings_dict.items()):
        # Get encodings
        dummy_input = torch.zeros(1, seq_length, d_model)
        with torch.no_grad():
            output = encoding(dummy_input)  # (1, seq_length, d_model)
            pe = output - dummy_input  # (1, seq_length, d_model)
            pe = pe.squeeze(0).cpu().numpy()  # (seq_length, d_model)
        
        # Heatmap
        im = axes[idx].imshow(pe.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        axes[idx].set_xlabel("Position")
        axes[idx].set_ylabel("Dimension")
        axes[idx].set_title(f"{name}")
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison metrics for multiple encoding methods.
    
    Args:
        results_dict: Dictionary mapping method names to generalization results.
        figsize: Figure size.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for name, results in results_dict.items():
        test_lengths = sorted(results["test_results"].keys())
        mean_diffs = [results["test_results"][l]["mean_diff"] for l in test_lengths]
        std_diffs = [results["test_results"][l]["std_diff"] for l in test_lengths]
        
        axes[0].plot(test_lengths, mean_diffs, marker="o", label=name, linewidth=2)
        axes[1].plot(test_lengths, std_diffs, marker="s", label=name, linewidth=2)
    
    axes[0].set_xlabel("Test Sequence Length")
    axes[0].set_ylabel("Mean Distribution Difference")
    axes[0].set_title("Mean Shift on Longer Sequences")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Test Sequence Length")
    axes[1].set_ylabel("Std Distribution Difference")
    axes[1].set_title("Std Shift on Longer Sequences")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig
