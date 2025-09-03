# import numpy as np

# def inject_empty_bin(hist, bin_idx):
#     hist.SetBinContent(bin_idx, 0)

# def inject_noisy_bin(hist, bin_idx, factor=5.0):
#     val = hist.GetBinContent(bin_idx)
#     hist.SetBinContent(bin_idx, val * factor)

# def inject_skew(hist, factor=1.5):
#     initial_norm = hist.Integral()
#     for i in range(1, hist.GetNbinsX() + 1):
#         shift = 1 + (i / hist.GetNbinsX()) * (factor - 1)
#         hist.SetBinContent(i, hist.GetBinContent(i) * shift)
#     hist.Scale(initial_norm / hist.Integral())  # Normalize back to initial integral

# def inject_secondary_peak(hist, pos, width, height):
#     initial_norm = hist.Integral()
#     for i in range(1, hist.GetNbinsX() + 1):
#         x = hist.GetBinCenter(i)
#         hist.SetBinContent(i, hist.GetBinContent(i) + height * hist.GetMaximum() * np.exp(-0.5 * ((x - pos)/width)**2))
#     hist.Scale(initial_norm / hist.Integral())  # Normalize back to initial integral


import numpy as np
from scipy.stats import norm, uniform

from .definitions import anomaly

def inject_empty_bin(hist_counts, bin_edges, bin_idx):
    """
    Set the content of a specific bin to zero.
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    bin_idx (int): Index of bin to empty (0-based)
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    new_hist[bin_idx] = 0
    return new_hist

def inject_noisy_bin(hist_counts, bin_edges, bin_idx, factor=5.0):
    """
    Multiply the content of a specific bin by a factor.
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    bin_idx (int): Index of bin to modify (0-based)
    factor (float): Multiplication factor
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    new_hist[bin_idx] *= factor
    return new_hist

def inject_skew(hist_counts, bin_edges, factor=1.5):
    """
    Apply a linear skew to the histogram.
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    factor (float): Skew factor
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    initial_norm = new_hist.sum()
    
    # Apply linear skew
    n_bins = len(new_hist)
    for i in range(n_bins):
        shift = 1 + (i / n_bins) * (factor - 1)
        new_hist[i] *= shift
    
    # Normalize back to initial integral
    if initial_norm > 0:
        new_hist *= initial_norm / new_hist.sum()
    
    return new_hist

def inject_shift_skew(hist_counts, bin_edges, skew_factor=1.5, tail_weight=1.0, shift=0.0):
    """
    Apply a configurable skew to the histogram with control over average shift and tail behavior.
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    skew_factor (float): Controls the degree of skew (1.0 = no change)
    tail_weight (float): Controls tail emphasis (>1 emphasizes tail, <1 reduces tail)
    shift (float): Shifts distribution left (<0) or right (>0) in units of bin width
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    initial_norm = new_hist.sum()
    
    if initial_norm == 0:
        return new_hist
    
    n_bins = len(new_hist)
    bin_centers = bin_edges
    
    # Normalize bin centers to [0,1] range for weighting
    normalized_pos = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    
    # Create skew weights
    skew_weights = np.exp(skew_factor * normalized_pos)
    skew_weights /= skew_weights.mean()  # Normalize mean to 1
    
    # Create tail weights (emphasize or reduce tail)
    tail_weights = normalized_pos**(tail_weight - 1.0)
    
    # Combine weights
    combined_weights = skew_weights * tail_weights
    
    # Apply shift (using linear interpolation)
    if shift != 0:
        bin_width = bin_edges[1] - bin_edges[0]
        shift_bins = shift / bin_width
        
        # Create shifted positions
        x = np.arange(n_bins)
        new_x = x - shift_bins
        
        # Interpolate
        from scipy.interpolate import interp1d
        interp_fn = interp1d(x, new_hist * combined_weights, 
                             bounds_error=False, fill_value=0.0)
        shifted_hist = interp_fn(new_x)
    else:
        shifted_hist = new_hist * combined_weights
    
    # Ensure non-negative
    shifted_hist = np.maximum(shifted_hist, 0)
    
    # Normalize back to initial integral
    if shifted_hist.sum() > 0:
        shifted_hist *= initial_norm / shifted_hist.sum()
    
    return shifted_hist

def inject_secondary_peak(hist_counts, bin_edges, pos, width, height):
    """
    Add a Gaussian peak to the histogram.
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    pos (float): Position of the new peak
    width (float): Width of the Gaussian
    height (float): Relative height (fraction of max bin content)
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    initial_norm = new_hist.sum()
    max_val = new_hist.max()
    
    # Calculate bin centers
    bin_centers = bin_edges
    
    # Add Gaussian peak
    gaussian = height * max_val * np.exp(-0.5 * ((bin_centers - pos)/width)**2)
    new_hist += gaussian
    
    # Normalize back to initial integral
    if initial_norm > 0:
        new_hist *= initial_norm / new_hist.sum()
    
    return new_hist

def inject_uniform_noise(hist_counts, bin_edges, noise_factor=0.1):
    """
    Add uniform noise to all bins.
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    noise_factor (float): Fraction of max count to use as noise amplitude
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    max_count = new_hist.max()
    noise = uniform.rvs(scale=noise_factor*max_count, size=len(new_hist))
    new_hist += noise
    return new_hist

def inject_sparse_bins(hist_counts, bin_edges, bin_chosen, factor=0.1):
    """
    Randomly reduce content of some bins (make them more empty).
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    fraction (float): Fraction of bins to affect
    factor (float): Reduction factor (0-1)
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    n_bins = len(new_hist)
    # n_to_modify = int(fraction * n_bins)
    bins_to_reduce = bin_chosen
    new_hist[bins_to_reduce] *= factor
    return new_hist

# def inject_phantom_peak(hist_counts, bin_edges, width=2.0, height=0.5):
#     """
#     Add a Gaussian peak at a random position that might mimic signal.
    
#     Parameters:
#     hist_counts (array): Bin counts from np.histogram
#     bin_edges (array): Bin edges from np.histogram
#     width (float): Width of the phantom peak
#     height (float): Height relative to maximum bin
    
#     Returns:
#     array: Modified histogram counts
#     """
#     new_hist = hist_counts.copy().astype("float32")
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     max_count = new_hist.max()
    
#     # Choose random position weighted by current emptiness
#     weights = 1 - (new_hist / (max_count + 1e-9))
#     pos = np.random.choice(bin_centers, p=weights/weights.sum())
    
#     # Add Gaussian
#     gaussian = height * max_count * np.exp(-0.5*((bin_centers-pos)/width)**2)
#     new_hist += gaussian
#     return new_hist

def inject_comb_artifact(hist_counts, bin_edges, spacing=5, factor=0.3):
    """
    Create regular pattern of suppressed bins (common in detector artifacts).
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    spacing (int): Every N-th bin is affected
    factor (float): Reduction factor for affected bins
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    new_hist[::spacing] *= factor
    return new_hist

def inject_edge_effects(hist_counts, bin_edges, region=0.1, factor=2.0):
    """
    Amplify bins near histogram edges (common in miscalibrations).
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    region (float): Fraction of bins at each end to affect
    factor (float): Amplification factor
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    n_bins = len(new_hist)
    n_edge = int(n_bins * region)
    
    # Amplify left and right edges
    new_hist[:n_edge] *= factor
    new_hist[-n_edge:] *= factor
    return new_hist

def inject_flat_background(hist_counts, bin_edges, level=0.2):
    """
    Add flat background to all bins (simulates contamination).
    
    Parameters:
    hist_counts (array): Bin counts from np.histogram
    bin_edges (array): Bin edges from np.histogram
    level (float): Fraction of average count to add
    
    Returns:
    array: Modified histogram counts
    """
    new_hist = hist_counts.copy().astype("float32")
    avg_count = new_hist.mean()
    new_hist += level * avg_count
    return new_hist


def zero_out_patch(image, center_x, center_y, patch_width, patch_height):
    """
    Sets a rectangular patch of the image to zero (black).
    Works with both grayscale (H x W) and RGB (H x W x 3) images.

    Parameters:
        image (numpy.ndarray): Input image (2D or 3D).
        center_x, center_y (float): Center coordinates in pixels.
        patch_width, patch_height (float): Size of the patch in pixels.

    Returns:
        numpy.ndarray: Modified image with the patch zeroed out.
    """
    img = image.copy().astype("float32")
    h, w = img.shape[:2]  # Image dimensions (height, width)

    # Calculate patch boundaries (clamped to image dimensions)
    x_min = int(max(0, center_x - patch_width // 2))
    x_max = int(min(w, center_x + patch_width // 2))
    y_min = int(max(0, center_y - patch_height // 2))
    y_max = int(min(h, center_y + patch_height // 2))

    img[y_min:y_max, x_min:x_max] = 0.0

    return img

def noise_in_patch(image, center_x, center_y, patch_width, patch_height,noise_factor):
    """
    Sets a rectangular patch of the image to zero (black).
    Works with both grayscale (H x W) and RGB (H x W x 3) images.

    Parameters:
        image (numpy.ndarray): Input image (2D or 3D).
        center_x, center_y (float): Center coordinates in pixels.
        patch_width, patch_height (float): Size of the patch in pixels.

    Returns:
        numpy.ndarray: Modified image with the patch zeroed out.
    """
    img = image.copy().astype("float32")
    h, w = img.shape[:2]  # Image dimensions (height, width)

    # Calculate patch boundaries (clamped to image dimensions)
    x_min = int(max(0, center_x - patch_width // 2))
    x_max = int(min(w, center_x + patch_width // 2))
    y_min = int(max(0, center_y - patch_height // 2))
    y_max = int(min(h, center_y + patch_height // 2))

    patch=img[y_min:y_max, x_min:x_max]

    mean_patch=patch.mean()
    max_patch=patch.max()
    std_patch=patch.std()

    noise=np.random.normal(mean_patch,std_patch,size=patch.shape)*noise_factor

    img[y_min:y_max, x_min:x_max]+=noise

    return img

def sparse_in_patch(image, center_x, center_y, patch_width, patch_height,sparse_factor):
    """
    Sets a rectangular patch of the image to zero (black).
    Works with both grayscale (H x W) and RGB (H x W x 3) images.

    Parameters:
        image (numpy.ndarray): Input image (2D or 3D).
        center_x, center_y (float): Center coordinates in pixels.
        patch_width, patch_height (float): Size of the patch in pixels.

    Returns:
        numpy.ndarray: Modified image with the patch zeroed out.
    """
    img = image.copy().astype("float32")
    h, w = img.shape[:2]  # Image dimensions (height, width)

    # Calculate patch boundaries (clamped to image dimensions)
    x_min = int(max(0, center_x - patch_width // 2))
    x_max = int(min(w, center_x + patch_width // 2))
    y_min = int(max(0, center_y - patch_height // 2))
    y_max = int(min(h, center_y + patch_height // 2))

    patch=img[y_min:y_max, x_min:x_max]

    # sparseness_distr=np.random.uniform(sparse_factor/2,sparse_factor,patch.shape)

    img[y_min:y_max, x_min:x_max]=patch*sparse_factor

    return img
