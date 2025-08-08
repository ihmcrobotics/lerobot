#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from lerobot.datasets.utils import load_image_as_numpy


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    for i in range(len(stats_list)):
        for fkey in stats_list[i]:
            for k, v in stats_list[i][fkey].items():
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                    )
                if v.ndim == 0:
                    raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                if k == "count" and v.shape != (1,):
                    raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")

def aggregate_feature_stats(stats_ft_list):
    """
    Combine per-episode stats for a single feature.

    Each element of stats_ft_list must be a dict with:
      - 'mean': array-like
      - 'std':  array-like
      - 'count' or 'n': int
    Optional:
      - 'min':  array-like
      - 'max':  array-like

    Returns:
      {
        'mean':  (D,) float64
        'std':   (D,) float64
        'count': int
        # 'min': (D,) float64  (if present in any input)
        # 'max': (D,) float64  (if present in any input)
      }
    """
    def _norm(x, name, idx):
        try:
            arr = np.asarray(x, dtype=np.float64)
        except Exception as e:
            raise TypeError(f"{name} at idx {idx} could not be converted to float64: {e}")
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr[None]
        return arr

    means, vars_, counts = [], [], []
    mins, maxs = [], []
    have_min = False
    have_max = False

    for i, s in enumerate(stats_ft_list):
        if s is None:
            continue

        if "mean" not in s or ("std" not in s):
            raise KeyError(f"Missing 'mean' or 'std' at idx {i}: keys={list(s.keys())}")

        m  = _norm(s["mean"], "mean", i)
        sd = _norm(s["std"],  "std",  i)

        # count key tolerance
        n = s.get("count", s.get("n", None))
        if n is None:
            raise KeyError(f"Missing 'count' (or 'n') at idx {i}")
        try:
            n = int(n)
        except Exception:
            raise TypeError(f"'count' must be int-like at idx {i}, got {type(n)}: {n}")

        if m.shape != sd.shape:
            raise ValueError(f"mean/std shape mismatch at idx {i}: {m.shape} vs {sd.shape}")

        means.append(m)
        vars_.append(sd ** 2)
        counts.append(n)

        if "min" in s and s["min"] is not None:
            mins.append(_norm(s["min"], "min", i))
            have_min = True
        if "max" in s and s["max"] is not None:
            maxs.append(_norm(s["max"], "max", i))
            have_max = True

    if not means:
        raise ValueError("No valid stats provided.")

    # Ensure all episodes have the same feature dimension
    shapes = {tuple(x.shape) for x in means}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent feature shapes across episodes: {shapes}. Fix upstream.")

    means = np.stack(means)        # (E, D)
    vars_ = np.stack(vars_)        # (E, D)
    counts = np.asarray(counts, dtype=np.int64)  # (E,)

    N = int(counts.sum())
    if N <= 1:
        # Degenerate case; fall back to simple average/std
        pooled_mean = means.mean(axis=0)
        pooled_std = np.sqrt(vars_.mean(axis=0))
    else:
        # Weighted (pooled) mean
        w = counts[:, None]                          # (E, 1)
        pooled_mean = (w * means).sum(axis=0) / N    # (D,)

        # Unbiased pooled variance:
        # SS_within = sum_i (n_i - 1) * var_i
        # SS_between = sum_i n_i * (mean_i - pooled_mean)^2
        ss_within = ((counts - 1)[:, None] * vars_).sum(axis=0)
        ss_between = (w * (means - pooled_mean) ** 2).sum(axis=0)
        denom = max(N - 1, 1)
        pooled_var = (ss_within + ss_between) / denom
        pooled_std = np.sqrt(np.maximum(pooled_var, 0.0))

    out = {
        "mean": pooled_mean.astype(np.float64),
        "std": pooled_std.astype(np.float64),
        "count": N,
    }

    # Optional min/max aggregation (elementwise)
    if have_min:
        # if some episodes lacked min/max, we only use the ones that exist
        mins = [m for m in mins if m is not None]
        # validate shapes
        shapes = {tuple(np.squeeze(np.asarray(m)).shape) for m in mins}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent 'min' shapes across episodes: {shapes}.")
        out["min"] = np.min(np.stack(mins), axis=0).astype(np.float64)

    if have_max:
        maxs = [m for m in maxs if m is not None]
        shapes = {tuple(np.squeeze(np.asarray(m)).shape) for m in maxs}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent 'max' shapes across episodes: {shapes}.")
        out["max"] = np.max(np.stack(maxs), axis=0).astype(np.float64)

    return out

def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats
