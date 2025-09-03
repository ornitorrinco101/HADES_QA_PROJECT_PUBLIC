# artificialDataset.py
import os
import random
import csv
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from .definitions import Distribution, anomaly  # Import from definitions.py

# Import your existing definitions
from .distributionDefinitions import *
from .anomaliesDefinitions import *

# === Configurable Parameters (can be set externally) === #
class DatasetConfig:
    def __init__(
        self,
        output_dir: str = "hist_dataset",
        n_total_per_type: int = 100,
        anomaly_fraction: float = 0.1,
        hist_bins: int = 50,
        parallel_workers: int = 4,
    ):
        self.output_dir = output_dir
        self.n_total_per_type = n_total_per_type
        self.anomaly_fraction = anomaly_fraction
        self.hist_bins = hist_bins
        self.parallel_workers = parallel_workers

# === Core Functions === #
def save_hist_as_numpy(hist_counts: np.ndarray, filepath: str) -> None:
    np.save(filepath, hist_counts.astype("float32"))

def generate_sample(
    dist: Distribution,
    counter: int,
    is_anomalous: bool = False,
    anomaly_loc: Optional[Dict] = None,
) -> tuple:
    """Generate a single sample (histogram) and return metadata."""
    data = dist.generate_data()
    hist_counts, bin_edges = np.histogram(data, **dist.hist_params)
    
    if is_anomalous and anomaly_loc:
        hist_counts = anomaly_loc["func"](hist_counts, bin_edges)
        filename = f"{dist.name}_{anomaly_loc['name']}_{counter:05d}.npy"
        label = 1
        anomaly_type = anomaly_loc["name"]
    else:
        filename = f"{dist.name}_{counter:05d}.npy"
        label = 0
        anomaly_type = "none"
    
    return filename, dist.name, anomaly_type, label, hist_counts

def process_distribution(
    dist: Distribution,
    config: DatasetConfig,
    counter_start: int = 0,
) -> List[tuple]:
    """Generate all samples for one distribution."""
    n_anomalies = int(config.anomaly_fraction * config.n_total_per_type)
    n_good = config.n_total_per_type - n_anomalies
    samples = []

    # Generate good samples
    for i in range(n_good):
        samples.append((dist, counter_start + i, False, None))
    
    # Generate anomalous samples
    for i in range(n_anomalies):
        anomaly_loc = random.choice(dist.anomalies)
        samples.append((dist, counter_start + n_good + i, True, anomaly_loc))
    
    return samples


def generate_dataset(config: DatasetConfig) -> None:
    """Main function to generate the full dataset."""
    os.makedirs(f"{config.output_dir}/npy", exist_ok=True)
    os.makedirs(f"{config.output_dir}/labels", exist_ok=True)
    
    # Prepare data structures
    all_histograms = []
    metadata = []

    distribution_configs = {
    "normal": Distribution(
        name="normal",
        data_generator=lambda: normal_dist(mean=1, sigma=1, size=10000),
        hist_params={"bins": config.hist_bins, "range": (0, 5)},
        anomalies=[
            anomaly("empty_bin", lambda h, e: inject_empty_bin(h, e, bin_idx=np.random.randint(0, 15))),
            anomaly("skew", lambda h, e: inject_skew(h, e, factor=4.0)),
            anomaly("secondary_peak", lambda h, e: inject_secondary_peak(h, e, 
                     pos=random.uniform(1.5, 4), 
                     width=random.uniform(0.1, 0.5), 
                     height=0.5)),
        ]
    ),
    "uniform": Distribution(
        name="uniform",
        data_generator=lambda: uniform_dist(low=3, high=7, size=10000),
        hist_params={"bins": config.hist_bins, "range": (3, 7)},
        anomalies=[
            anomaly("noisy_bin", lambda h, e: inject_noisy_bin(h, e, bin_idx=np.random.randint(20, 40), factor=5.0)),
        ]
    ),
    "double_peak": Distribution(
        name="double_peak",
        data_generator=lambda: double_peak(mean1=1, sigma1=0.5, mean2=3, sigma2=0.4, frac=0.6, size=10000),
        hist_params={"bins": config.hist_bins, "range": (0, 5)},
        anomalies=[
            anomaly("empty_bin", lambda h, e: inject_empty_bin(h, e, bin_idx=np.random.randint(10, 20))),
            anomaly("skew", lambda h, e: inject_skew(h, e, factor=4.0)),
        ]
    )
    }
    
    # Generate all samples
    for dist in distribution_configs.values():
        samples = process_distribution(dist, config, len(all_histograms))
        with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
            for filename, dist_name, anomaly_type, label, hist_counts in executor.map(
                lambda args: generate_sample(*args), samples
            ):
                all_histograms.append(hist_counts)
                metadata.append([filename, dist_name, anomaly_type, label])
    
    # Save single numpy array
    combined_file = f"{config.output_dir}/npy/histograms_{config.n_total_per_type}_hpt.npy"
    np.save(combined_file, np.array(all_histograms, dtype=np.float32))
    
    # Save metadata CSV
    labels_file = f"{config.output_dir}/labels/labels_{config.n_total_per_type}_hpt.csv"
    if os.path.exists(labels_file):
        os.remove(labels_file)
    
    with open(labels_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "distribution", "anomaly_type", "label"])
        for idx, (filename, dist_name, anomaly_type, label) in enumerate(metadata):
            writer.writerow([idx, dist_name, anomaly_type, label])
    
    print(f"Generated {len(all_histograms)} histograms in {combined_file}")

# def generate_dataset(config: DatasetConfig) -> None:
#     """Main function to generate the full dataset."""
#     os.makedirs(f"{config.output_dir}/npy/{config.n_total_per_type}_hpt", exist_ok=True)
#     os.makedirs(f"{config.output_dir}/labels", exist_ok=True)
    
#     # Clear existing files
#     for f in os.listdir(f"{config.output_dir}/npy/{config.n_total_per_type}_hpt"):
#         os.remove(os.path.join(f"{config.output_dir}/npy/{config.n_total_per_type}_hpt", f))

#     # Prepare distributions (modify hist_params to use config.bins if needed)
#     distribution_configs = {
#     "normal": Distribution(
#         name="normal",
#         data_generator=lambda: normal_dist(mean=1, sigma=1, size=10000),
#         hist_params={"bins": config.hist_bins, "range": (0, 5)},
#         anomalies=[
#             anomaly("empty_bin", lambda h, e: inject_empty_bin(h, e, bin_idx=np.random.randint(0, 15))),
#             anomaly("skew", lambda h, e: inject_skew(h, e, factor=4.0)),
#             anomaly("secondary_peak", lambda h, e: inject_secondary_peak(h, e, 
#                      pos=random.uniform(1.5, 4), 
#                      width=random.uniform(0.1, 0.5), 
#                      height=0.5)),
#         ]
#     ),
#     "uniform": Distribution(
#         name="uniform",
#         data_generator=lambda: uniform_dist(low=3, high=7, size=10000),
#         hist_params={"bins": config.hist_bins, "range": (3, 7)},
#         anomalies=[
#             anomaly("noisy_bin", lambda h, e: inject_noisy_bin(h, e, bin_idx=np.random.randint(20, 40), factor=5.0)),
#         ]
#     ),
#     "double_peak": Distribution(
#         name="double_peak",
#         data_generator=lambda: double_peak(mean1=1, sigma1=0.5, mean2=3, sigma2=0.4, frac=0.6, size=10000),
#         hist_params={"bins": config.hist_bins, "range": (0, 5)},
#         anomalies=[
#             anomaly("empty_bin", lambda h, e: inject_empty_bin(h, e, bin_idx=np.random.randint(10, 20))),
#             anomaly("skew", lambda h, e: inject_skew(h, e, factor=4.0)),
#         ]
#     )
#     }

#     # Generate all samples in parallel
#     all_samples = []
#     counter = 0
#     for dist in distribution_configs.values():
#         samples = process_distribution(dist, config, counter)
#         all_samples.extend(samples)
#         counter += config.n_total_per_type

#     # Parallel execution
#     with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
#         results = list(executor.map(
#             lambda args: generate_sample(*args), 
#             all_samples
#         ))

#     # Write CSV and save NPY files
#     labels_file=f"{config.output_dir}/labels/labels_{config.n_total_per_type}_hpt.csv"
#     if os.path.exists(labels_file):
#         os.remove(labels_file)  # Delete the old file
#     with open(labels_file, "w", newline="") as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(["filename", "distribution", "anomaly_type", "label"])
        
#         for filename, dist_name, anomaly_type, label, hist_counts in results:
#             save_hist_as_numpy(hist_counts, f"{config.output_dir}/npy/{config.n_total_per_type}_hpt/{filename}")
#             csv_writer.writerow([filename, dist_name, anomaly_type, label])

#     print(f"Generated {len(results)} histograms in {config.output_dir} ({config.anomaly_fraction*len(results):.0f} anomalies)")

# # === Usage Example === #
# if __name__ == "__main__":
#     config = DatasetConfig(
#         output_dir="my_dataset",
#         n_total_per_type=500,  # 500 histograms per distribution
#         anomaly_fraction=0.2,  # 20% anomalies
#         hist_bins=30,          # 30 bins per histogram
#         parallel_workers=8,    # Use 8 CPU cores
#     )
#     generate_dataset(config)























# import os
# import random
# import csv
# import numpy as np
# import ROOT


# ROOT.gErrorIgnoreLevel = ROOT.kWarning

# from distributionDefinitions import *
# from anomaliesDefinitions import *

# ROOT.gROOT.SetBatch(True)  # Avoid GUI pop-ups

# # Struct for anomalies that can hold the name and the function with lambda in the future
# def anomaly(name, func):
#     return {"name": name, "func": func}

# # Class to encapsulate distribution and its anomalies
# class Distribution:
#     def __init__(self, name, data_generator, hist_params, anomalies):
#         self.name = name
#         self.generate_data = data_generator
#         self.hist_params=hist_params
#         self.anomalies = anomalies


# def fill_TH1(data, name, nbins, xmin, xmax):
#     return np.histogram(data, bins=nbins, range=(xmin, xmax))

# def fill_TH2(data_2d, name, xbins, xmin, xmax, ybins, ymin, ymax):
#     h = ROOT.TH2F(name, name, xbins, xmin, xmax, ybins, ymin, ymax)
#     for x, y in data_2d:
#         h.Fill(x, y)
#     return h


# # === Define available distributions and associated anomalies === #
# distribution_configs = {
#     "normal": Distribution(
#         name="normal",
#         data_generator=lambda: normal_dist(mean=1, sigma=1, size=10000),
#         hist_params={"bins": 50, "range": (0, 5)},
#         anomalies=[
#             anomaly("empty_bin", lambda h, e: inject_empty_bin(h, e, bin_idx=np.random.randint(0, 15))),
#             anomaly("skew", lambda h, e: inject_skew(h, e, factor=4.0)),
#             anomaly("secondary_peak", lambda h, e: inject_secondary_peak(h, e, 
#                      pos=random.uniform(1.5, 4), 
#                      width=random.uniform(0.1, 0.5), 
#                      height=0.5)),
#         ]
#     ),
#     "uniform": Distribution(
#         name="uniform",
#         data_generator=lambda: uniform_dist(low=3, high=7, size=10000),
#         hist_params={"bins": 50, "range": (3, 7)},
#         anomalies=[
#             anomaly("noisy_bin", lambda h, e: inject_noisy_bin(h, e, bin_idx=np.random.randint(20, 40), factor=5.0)),
#         ]
#     ),
#     "double_peak": Distribution(
#         name="double_peak",
#         data_generator=lambda: double_peak(mean1=1, sigma1=0.5, mean2=3, sigma2=0.4, frac=0.6, size=10000),
#         hist_params={"bins": 50, "range": (0, 5)},
#         anomalies=[
#             anomaly("empty_bin", lambda h, e: inject_empty_bin(h, e, bin_idx=np.random.randint(10, 20))),
#             anomaly("skew", lambda h, e: inject_skew(h, e, factor=4.0)),
#         ]
#     )
# }


# # === Parameters === #
# output_dir = "hist_dataset"
# n_total_per_type = 100
# anomaly_fraction = 0.1


# def save_hist_as_image(hist, filepath):
#     canvas = ROOT.TCanvas()
#     hist.Draw()
#     canvas.SaveAs(filepath)
#     canvas.Close()

# def save_hist_as_numpy(hist_counts, filepath):
#     np.save(filepath, hist_counts.astype("float32"))




# images_dir = f"{output_dir}/images"
# numpy_dir = f"{output_dir}/npy"
# # === Create output directories if they do not exist === #
# os.makedirs(numpy_dir, exist_ok=True)
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(images_dir, exist_ok=True)

# # === Clear images folder if it exists === #
# images_dir = f"{output_dir}/npy"

# if os.path.exists(f"{output_dir}/npy"):
#     for f in os.listdir(f"{output_dir}/npy"):
#         os.remove(os.path.join(f"{output_dir}/npy", f))

# # === Dataset generation loop === #
# counter = 0
# with open(f"{output_dir}/labels.csv", "w", newline="") as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["filename", "distribution", "anomaly_type", "label"])
    
#     counter = 0
#     for dist_key, dist in distribution_configs.items():
#         n_anomalies = int(anomaly_fraction * n_total_per_type)
#         n_good = n_total_per_type - n_anomalies

#         print(f"Generating {dist.name} — {n_good} good + {n_anomalies} anomalous")

#         # Generate good samples
#         for _ in range(n_good):
#             data = dist.generate_data()
#             hist_counts, bin_edges = np.histogram(data, **dist.hist_params)
            
#             filename = f"{dist.name}_{counter:05d}.npy"
#             save_hist_as_numpy(hist_counts, f"{output_dir}/npy/{filename}")
#             csv_writer.writerow([filename, dist.name, "none", 0])
#             counter += 1

#         # Generate anomalous samples
#         for _ in range(n_anomalies):
#             data = dist.generate_data()
#             hist_counts, bin_edges = np.histogram(data, **dist.hist_params)
            
#             anomaly_loc = random.choice(dist.anomalies)
#             hist_counts = anomaly_loc["func"](hist_counts, bin_edges)  # Capture returned value
            
#             filename = f"{dist.name}_{anomaly_loc['name']}_{counter:05d}.npy"
#             save_hist_as_numpy(hist_counts, f"{output_dir}/npy/{filename}")
#             csv_writer.writerow([filename, dist.name, anomaly_loc["name"], 1])
#             counter += 1

# print(f"Dataset generated in '{output_dir}' with {counter} total histograms.")