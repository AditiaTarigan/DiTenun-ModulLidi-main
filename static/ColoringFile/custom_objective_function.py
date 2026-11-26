import numpy as np
from itertools import product

def calculate_user_color_preferences(hsv_img):
    user_prefs = np.array([[18, 75, 44], [359, 91, 55]], dtype=np.int32)
    if hsv_img is None:
        return 0.0
    img = np.asarray(hsv_img)
    if img.ndim != 3 or img.shape[2] < 3:
        return 0.0
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h = img[:, :, 0].astype(np.int32)
    s = img[:, :, 1].astype(np.int32)
    v = img[:, :, 2].astype(np.int32)
    h360 = ((h.astype(np.int32) % 180) * 2).astype(np.int32)
    flat = np.stack([h360.ravel(), s.ravel(), v.ravel()], axis=1)
    unique_colors = np.unique(flat, axis=0).astype(np.int32)
    if unique_colors.shape[0] == 0:
        return 0.0
    prefs = user_prefs.copy()
    prefs[:, 0] = (prefs[:, 0].astype(np.int32) % 360)
    diffs_h = np.abs(unique_colors[:, None, 0] - prefs[None, :, 0])
    diffs_h = np.minimum(diffs_h, 360 - diffs_h)
    diffs_s = np.abs(unique_colors[:, None, 1] - prefs[None, :, 1])
    diffs_v = np.abs(unique_colors[:, None, 2] - prefs[None, :, 2])
    dist = np.sqrt((diffs_h / 180.0) ** 2 + (diffs_s / 255.0) ** 2 + (diffs_v / 255.0) ** 2)
    min_dist_per_color = np.min(dist, axis=1)
    fitness_per_color = np.exp(-min_dist_per_color)
    total_pixels = h.size
    color_counts = {}
    for i in range(unique_colors.shape[0]):
        key = tuple(int(x) for x in unique_colors[i])
        color_counts[key] = 0
    keys_arr = unique_colors
    idx = np.ravel_multi_index(flat.T, (flat[:, 0].max() + 1, 256, 256)) if flat.size else np.array([])
    uniq_idx, counts = np.unique(idx, return_counts=True)
    weights = counts.astype(np.float64)
    fitness_weighted = np.sum(fitness_per_color * (weights / np.sum(weights)))
    final_fitness = 0.5 * np.mean(fitness_per_color) + 0.5 * fitness_weighted
    if np.array_equal(np.sort(unique_colors, axis=0), np.sort(prefs, axis=0)):
        final_fitness = 1.0
    return float(np.clip(final_fitness, 0.0, 1.0))