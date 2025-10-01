"""
Additional utility functions.
"""
from typing import Dict, Any, Tuple
import numpy as np
import json
import os


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a file saved by save_data.
    Returns (obj, metadata_dict_or_empty).
    """
    base, ext = os.path.splitext(path)
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        meta = {}
        meta_path = base + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return arr, meta

    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        # convert to dict of arrays
        arrays = {k: data[k] for k in data.files}
        meta = {}
        meta_path = base + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return arrays, meta

    if ext == ".json":
        with open(path) as f:
            obj = json.load(f)
        meta = {}
        meta_path = base + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return obj, meta

    raise ValueError(f"Unsupported extension {ext}. Use .npy/.npz/.json")

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def save_data(path: str, obj: Any, metadata: Dict = None, compress: bool = True) -> str:
    """
    Save `obj` to disk. Behavior chosen by file extension or object type.
    - path ending in .npy  -> saves a single numpy array
    - path ending in .npz  -> saves a dict of arrays (or name provided)
    - path ending in .json -> saves lists/dicts of primitives
    - otherwise: will choose .npy/.npz/.json based on obj type

    Returns the full path written.
    """
    # choose extension if not provided
    base, ext = os.path.splitext(path)
    if ext == "":
        # pick sensible default
        if isinstance(obj, np.ndarray):
            ext = ".npy"
            path = base + ext
        elif isinstance(obj, (list, dict)):
            ext = ".json"
            path = base + ext
        else:
            raise ValueError("Provide path with extension or pass ndarray/list/dict.")

    _ensure_dir(path)

    if ext == ".npy":
        if not isinstance(obj, np.ndarray):
            raise TypeError(".npy requires a numpy.ndarray")
        # save array (dtype & shape preserved)
        np.save(path, obj)
        # optionally write meta alongside
        if metadata:
            meta_path = base + ".meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        return path

    if ext == ".npz":
        # if user provides a single array, save as array=np
        if isinstance(obj, np.ndarray):
            # save under key 'arr_0'
            arrays = {"arr_0": obj}
        elif isinstance(obj, dict):
            arrays = obj
        elif isinstance(obj, list):
            # save list of arrays if arrays; otherwise wrap it into a dict
            if all(isinstance(x, np.ndarray) for x in obj):
                arrays = {f"arr_{i}": x for i, x in enumerate(obj)}
            else:
                # convert to json instead
                raise TypeError("npz requires arrays or dict-of-arrays. Use .json for general lists.")
        else:
            raise TypeError("npz requires ndarray(s) or dict-of-arrays.")

        if compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)

        if metadata:
            meta_path = base + ".meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        return path

    if ext == ".json":
        # only for lists/dicts with primitive types (int/float/str/bool)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        if metadata:
            meta_path = base + ".meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        return path

    raise ValueError(f"Unsupported extension {ext}. Use .npy/.npz/.json or provide a numpy array/dict/list.")


def compute_spike_matrix (spikeTimes: np.ndarray, spikeClusters: np.ndarray, time: np.ndarray, start_time: float,
                           end_time: float, sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spike matrix from spike times and clusters"""

    timestamps = np.arange(time[0],math.ceil(time[-1]), 0.025) # get timestamps in 25 ms bins
    cell_IDs = np.unique(spikeClusters) # get number of cells
    spike_matrix = np.zeros((len(cell_IDs), len(timestamps))) # create matrix with zeros
    n_bins = len(timestamps) 

    # Fill matrix with spike count for each neuron per timebin
    # then iterate through number of place cells in place cells list
    for counter, cell in enumerate(cell_IDs):

        cell_indexes = np.where(spikeClusters == cell) # find the indices of the spikes of that cell
        spike_times_OE = spikeTimes[cell_indexes] # extract times at which that cell fired

        # need to convert times from Openephys times to seconds or milliseconds
        spike_times = spike_times_OE / sampling_rate # divide by sampling rate to get seconds

        # Extract spikes from the recording session (only matters if there is multiple sessions in one recording)
        spike_times_session = spike_times[(spike_times >= start_time) & (spike_times <= end_time)] # extract spikes from that session
        
        # then allocate 1s in the matrix in the timebins when it fired - use histogram function
        binnedSpikes, _= np.histogram(spike_times_session, bins = n_bins)
        zBinnedSpikes = stats.zscore(binnedSpikes) # z-score the binned spikes
        spike_matrix[counter, :] = zBinnedSpikes # fill the matrix with the z-scored binned spikes

    # Remove rows with all NaNs
    spike_matrix_clean = spike_matrix
    cell_IDs_clean = cell_IDs
    row = 0
    for it in range(spike_matrix.shape[0]):
        if np.isnan(spike_matrix_clean[row,:]).all() == True:
            spike_matrix_clean = np.delete(spike_matrix_clean, row, 0)
            cell_IDs_clean = np.delete(cell_IDs_clean , row, 0)
            row = row
        else:
            row = row + 1
        
    return spike_matrix_clean, cell_IDs_clean

def rebin_spikes(spike_matrix: np.ndarray, old_dt: float, new_dt: float) -> np.ndarray:
    """
    Re-bin a spike matrix from old_dt to new_dt.

    Parameters
    ----------
    spike_matrix : np.ndarray. Array of shape (n_neurons, n_times) with binary spike indicators or counts.
    old_dt : float. Original bin width in seconds.
    new_dt : float. Desired bin width in seconds.

    Returns
    -------
    rebinned : np.ndarray. Array of shape (n_neurons, n_new_times) with rebinned spike counts.
    """
    factor = int(round(new_dt / old_dt))
    if abs(new_dt / old_dt - factor) > 1e-8:
        raise ValueError("new_dt must be an integer multiple of old_dt")

    n_neurons, n_times = spike_matrix.shape
    n_new_times = n_times // factor

    # truncate extra timepoints if not divisible
    trimmed = spike_matrix[:, :n_new_times * factor]

    # reshape and sum within bins
    rebinned = trimmed.reshape(n_neurons, n_new_times, factor).sum(axis=2)
    return rebinned

