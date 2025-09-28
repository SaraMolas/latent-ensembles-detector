"""
Plotting functions for latent ensembles detector.
"""
# src/place_cell_simulations/viz.py
from typing import List, Sequence, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import os

def _ensure_dir_for_file(path: str) -> None:
    """Internal: ensure directory exists for a given file path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def plot_principal_cells_heatmap(
    principal_cells: Sequence[Sequence[int]],
    n_neurons: Optional[int] = None,
    sort_by: Optional[str] = None,
    figsize: Optional[tuple] = (8, 6),
    xlabel: Optional[str] = "Neurons (id)",
    ylabel: Optional[str] = "Ensemble",
    title: Optional[str] = "Principal cells per ensemble",
    cmap: Optional[str] = "viridis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a binary heatmap (ensembles x neurons) indicating which neurons are
    principal for each ensemble.

    Args:
        principal_cells: list of lists; principal_cells[i] are neuron ids for ensemble i.
        n_neurons: total number of neurons. If None, inferred from max id + 1.
        sort_by: None | 'size' | 'first_index'
            - 'size' sorts assemblies by number of principal cells (descending)
            - 'first_index' sorts by the first principal neuron id
        figsize: figure size
        save_path: optional; if provided, saves figure to path
    Returns:
        (fig, ax)
    """
    # infer n_neurons
    if n_neurons is None:
        max_id = -1
        for lst in principal_cells:
            if len(lst) > 0:
                max_id = max(max_id, max(lst))
        n_neurons = max_id + 1 if max_id >= 0 else 0

    n_ensembles= len(principal_cells)
    mat = np.zeros((n_ensembles, n_neurons), dtype=int)
    for i, lst in enumerate(principal_cells):
        for neuron in lst:
            if 0 <= neuron < n_neurons:
                mat[i, neuron] = 1

    # optional sorting of rows (assemblies)
    order = np.arange(n_ensembles)
    if sort_by == "size":
        sizes = np.sum(mat, axis=1)
        order = np.argsort(-sizes)  # descending
    elif sort_by == "first_index":
        first_idx = np.array([min((lst) or [n_neurons]) for lst in principal_cells])
        order = np.argsort(first_idx)

    mat_sorted = mat[order, :]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat_sorted, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_yticks(np.arange(n_ensembles))
    ax.set_yticklabels((order + 0).tolist())  # optionally show original assembly indices
    ax.set_xlim(-0.5, n_neurons - 0.5 if n_neurons>0 else 0.5)
    fig.colorbar(im, ax=ax, label="principal (1) / not (0)")

    if save_path:
        _ensure_dir_for_file(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, ax