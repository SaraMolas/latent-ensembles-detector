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


