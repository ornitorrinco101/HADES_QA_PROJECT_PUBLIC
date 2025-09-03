# artificialDataset.py
import os
import random
import csv
import numpy as np
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import awkward as ak
import yaml
import re
import pandas as pd

# Import your existing definitions
from .distributionDefinitions import *
from .anomaliesDefinitions import *
from .definitions import Distribution, anomaly  # Import from definitions.py

# root_dir = Path(__file__).parents[2] # Goes up 2 levels to ANOMALYDETECTIONHADES
# sys.path.append(str(root_dir))
# print(root_dir)

# === Configurable Parameters (can be set externally) === #
class DatasetConfig:
    def __init__(
        self,
        output_dir: str = "HADES_t1_dataset",
        n_total_per_type: int = 100,
        anomaly_fraction: float = 0.1,
        hist_bins: int = 50,
        parallel_workers: int = 4,
        distribution_configs: Dict[str, Distribution] = None,
        config_path: str = None,  # Path to YAML config
        root_dir: str = None      # Root directory for path substitution
    ):
        self.output_dir = output_dir
        self.n_total_per_type = n_total_per_type
        self.anomaly_fraction = anomaly_fraction
        self.hist_bins = hist_bins
        self.parallel_workers = parallel_workers
        self.config_path = config_path
        self.root_dir = root_dir
        
        # Load configs if path provided, else use provided or default
        if config_path:
            self.distribution_configs = load_distribution_configs(config_path, root_dir)
        elif distribution_configs:
            self.distribution_configs = distribution_configs
        else:
            print("Fallback to prev dictionary")
            self.distribution_configs = self._default_distributions()
    
    def _default_distributions(self):
        # Fallback to hardcoded configs
        return {
            "START_beam_mod1": Distribution(
                name="START_beam_mod1",
                data_generator=lambda: START_beam_profile(mod=1),
                hist_params={"bins": 20},  # Explicit bin edges 0-20
                anomalies=[
                    anomaly("noisy_bin", lambda h, e: inject_noisy_bin(
                        h, e, 
                        bin_idx=np.random.choice(np.where(h > 0)[0]),  # Choose non-empty bin
                        factor=np.random.uniform(5, 10))),
                    anomaly("empty_bin", lambda h, e: inject_empty_bin(
                        h, e, 
                        bin_idx=np.random.choice(np.where(h > 0)[0]))),  # Choose non-empty bin
                    anomaly("uniform_noise", lambda h, e: inject_uniform_noise(
                        h, e, 
                        noise_factor=np.random.uniform(0.05, 0.2))),
                    anomaly("sparse_bin", lambda h, e: inject_sparse_bins(
                        h, e, 
                        bin_chosen=np.random.choice(np.where(h > 0)[0]),  # Choose non-empty bin
                        factor=np.random.uniform(0.1, 0.7))),
                #     anomaly("comb_artifact", lambda h, e: inject_comb_artifact(
                #         h, e, 
                #         spacing=np.random.randint(2, 4),
                #         factor=np.random.uniform(0.3, 0.5)))
                # 
                ]
            ),
            "START_beam_mod2": Distribution(
                name="START_beam_mod2",
                data_generator=lambda: START_beam_profile(mod=2),
                hist_params={"bins": 20},  # Explicit bin edges 0-20
                anomalies=[
                    anomaly("noisy_bin", lambda h, e: inject_noisy_bin(
                        h, e, 
                        bin_idx=np.random.choice(np.where(h > 0)[0]),  # Choose non-empty bin
                        factor=np.random.uniform(5, 10))),
                    anomaly("empty_bin", lambda h, e: inject_empty_bin(
                        h, e, 
                        bin_idx=np.random.choice(np.where(h > 0)[0]))),  # Choose non-empty bin
                    anomaly("uniform_noise", lambda h, e: inject_uniform_noise(
                        h, e, 
                        noise_factor=np.random.uniform(0.05, 0.2))),
                    anomaly("sparse_bin", lambda h, e: inject_sparse_bins(
                        h, e, 
                        bin_chosen=np.random.choice(np.where(h > 0)[0]),  # Choose non-empty bin
                        factor=np.random.uniform(0.1, 0.7))),
                    anomaly("comb_artifact", lambda h, e: inject_comb_artifact(
                        h, e, 
                        spacing=np.random.randint(2, 4),
                        factor=np.random.uniform(0.3, 0.5)))
                ]
            ),
            "RICH_trends": Distribution(
                name="RICH_trends",
                data_generator=lambda: RICH_total_mult(minval=100, maxval=140, averaging_param=5, numbins=51)[1],
                hist_params={"bins": 50},
                anomalies=[
                    anomaly("skew", lambda h, e: inject_skew(
                        h, e, 
                        factor=np.random.choice([np.random.uniform(0.5, 0.9), np.random.uniform(1.1, 1.5)]))
                    ),
                    anomaly("noisy_bin", lambda h, e: inject_noisy_bin(
                        h, e,
                        bin_idx=np.random.choice(np.where(h > 0)[0]),
                        factor=np.random.uniform(1.2, 2.0))
                    ),
                    anomaly("sparse_bin", lambda h, e: inject_sparse_bins(
                        h, e,
                        bin_chosen=np.random.choice(np.where(h > 0)[0]),
                        factor=np.random.uniform(0.1, 0.8))
                    ),
                    anomaly("comb_artifact", lambda h, e: inject_comb_artifact(
                        h, e,
                        spacing=np.random.randint(2, 11),
                        factor=np.random.uniform(0.1, 0.8))
                    ),
                    anomaly("edge_effects", lambda h, e: inject_edge_effects(
                        h, e,
                        region=np.random.uniform(0.1, 0.4),
                        factor=np.random.uniform(1.2, 1.5))
                    )
                ]
            ),
            "RICH_CalsXY": Distribution(
                name="RICH_CalsXY",
                data_generator=lambda: RICH_CalsXY(root_dir=root_dir, og_size=572, kernel_downsample=6, 
                                                add_noise=True, noise_factor=0.2, lmax=0.25),
                hist_params={},  # No histogramming for images
                anomalies=[
                    anomaly("empty_patch", lambda img, _: zero_out_patch(
                        img,
                        center_x=int(np.random.uniform(40//6, 530//6)),
                        center_y=int(np.random.uniform(70//6, 500//6)),
                        patch_width=30//6,
                        patch_height=50//6)
                    ),
                    anomaly("noisy_patch", lambda img, _: noise_in_patch(
                        img,
                        center_x=int(np.random.uniform(40//6, 530//6)),
                        center_y=int(np.random.uniform(70//6, 500//6)),
                        patch_width=30//6,
                        patch_height=50//6,
                        noise_factor=np.random.uniform(0.2, 0.6))
                    ),
                    anomaly("sparse_patch", lambda img, _: sparse_in_patch(
                        img,
                        center_x=int(np.random.uniform(40//6, 530//6)),
                        center_y=int(np.random.uniform(70//6, 500//6)),
                        patch_width=30//6,
                        patch_height=50//6,
                        sparse_factor=np.random.uniform(0.2, 0.8))
                    )
                ]
            ),
            "ToF_multiplicity": Distribution(
                name="ToF_multiplicity",
                data_generator=lambda: ToF_multiplicity(nsamples=4000, numbins=100)[1],
                hist_params={"bins": 100},
                anomalies=[
                    anomaly("uniform_noise", lambda h, e: inject_uniform_noise(
                        h, e,
                        noise_factor=np.random.uniform(0.01, 0.4))
                    ),
                    anomaly("skew", lambda h, e: inject_shift_skew(
                        h, e,
                        skew_factor=np.random.choice([np.random.uniform(-20, -5), np.random.uniform(10, 20)]),
                        tail_weight=1.0,
                        shift=0.0)
                    ),
                    anomaly("comb_artifact", lambda h, e: inject_comb_artifact(
                        h, e,
                        spacing=np.random.randint(2, 11),
                        factor=np.random.uniform(0.3, 0.6))
                    )
                ]
            ),
            "ToF_sum": Distribution(
                name="ToF_sum",
                data_generator=lambda: ToF_sum(nsamples=6000, noise_factor=0.15, numbins=150)[1],
                hist_params={"bins": 150},
                anomalies=[
                    anomaly("uniform_noise", lambda h, e: inject_uniform_noise(
                        h, e,
                        noise_factor=np.random.uniform(0.15, 0.4))
                    ),
                    anomaly("skew", lambda h, e: inject_shift_skew(
                        h, e,
                        skew_factor=np.random.choice([np.random.uniform(-30, -10), np.random.uniform(1.1, 3.0)]),
                        tail_weight=1.0,
                        shift=0.0)
                    ),
                    anomaly("secondary_peak", lambda h, e: inject_secondary_peak(
                        h, e,
                        pos=np.random.randint(0, 10),
                        width=np.random.randint(1, 4),
                        height=h.mean() * np.random.uniform(0.003, 0.006))
                    )
                ]
            ),
        }  # Your original hardcoded dictionary

# === Core Functions === #
def save_hist_as_numpy(hist_counts: np.ndarray, filepath: str) -> None:
    np.save(filepath, hist_counts.astype("float32"))

def generate_sample(dist: Distribution, counter: int, is_anomalous: bool = False,
                   anomaly_loc: Optional[Dict] = None) -> tuple:
    """Generate a single sample and return metadata."""
    data = dist.generate_data()
    
    # Handle different data types
    if dist.name == "RICH_CalsXY":
        # 2D image data - no histogramming
        processed_data = data
        bin_edges = None
    else:
        # 1D histogram data
        if isinstance(data, tuple):  # (bins, counts)
            hist_counts, bin_edges = data[1], data[0]
        else:
            hist_counts, bin_edges = np.histogram(data, **dist.hist_params)
        processed_data = hist_counts
    
    # Apply anomaly if needed
    if is_anomalous and anomaly_loc:
        if dist.name == "RICH_CalsXY":
            processed_data = dist.second_downsampler(anomaly_loc["func"](processed_data, None))  # No bin edges for images
        else:
            # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            processed_data = anomaly_loc["func"](processed_data, bin_edges)

        filename = f"{dist.name}_{anomaly_loc['name']}_{counter:05d}"
        label = 1
        anomaly_type = anomaly_loc["name"]
    else:
        if dist.name=="RICH_CalsXY":
            processed_data=dist.second_downsampler(processed_data)
        filename = f"{dist.name}_{counter:05d}"
        label = 0
        anomaly_type = "none"
    
    return filename, dist.name, anomaly_type, label, processed_data

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
from .definitions import Distribution, anomaly

def load_distribution_configs(yaml_path: str, root_dir) -> Dict[str, Distribution]:
    """Load distribution configurations from YAML file with proper error handling."""
    try:
        with open(str(root_dir)+yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")
    
    distributions = {}
    
    for dist_name, dist_config in config_data.get('distributions', {}).items():
        try:
            # Handle root_dir substitution in data generator string
            data_gen_str = dist_config['data_generator']
            if root_dir and 'root_dir' in data_gen_str:
                data_gen_str = data_gen_str.replace('root_dir', f"'{root_dir}'")
            
            # Safely evaluate the lambda function
            data_generator = eval(data_gen_str)
            second_downsampler = None
            if 'second_downsampler' in dist_config:
                second_downsampler = eval(dist_config['second_downsampler'])
            
            # Parse anomalies
            anomalies = []
            for anomaly_config in dist_config.get('anomalies', []):
                anomaly_func = eval(anomaly_config['function'])
                anomalies.append(anomaly(anomaly_config['name'], anomaly_func))
            
            # Create Distribution object
            distributions[dist_name] = Distribution(
                name=dist_config['name'],
                data_generator=data_generator,
                hist_params=dist_config.get('hist_params', {}),
                anomalies=anomalies,
                second_downsampler=second_downsampler
            )
            
        except KeyError as e:
            print(f"Warning: Missing key {e} in distribution {dist_name}, skipping")
        except Exception as e:
            print(f"Warning: Error loading distribution {dist_name}: {e}, skipping")
    
    return distributions


def generate_dataset(config: DatasetConfig) -> None:
    """Main function to generate the full dataset."""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/numpy", exist_ok=True)
    os.makedirs(f"{config.output_dir}/labels", exist_ok=True)
    
    # Load distribution configs from YAML if provided, else use default
    if hasattr(config, 'config_path') and config.config_path:
        distribution_configs = load_distribution_configs(config.config_path, config.root_dir)
    else:
        distribution_configs = config.distribution_configs
    
    all_data = {}
    metadata = []

    n_events = config.n_total_per_type
    n_anomalies_total = int(n_events * config.anomaly_fraction)
    
    # For each distribution, determine anomaly pattern
    anomaly_patterns = {}
    for dist_name in distribution_configs.keys():
        anomaly_indices = random.sample(range(n_events), n_anomalies_total)
        anomaly_patterns[dist_name] = [i in anomaly_indices for i in range(n_events)]

    # Initialize data storage for each detector
    for dist_name in distribution_configs.keys():
        all_data[dist_name] = []

    # Generate all samples
    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        for event_idx in range(n_events):
            event_metadata = {"event_id": f"event_{event_idx:05d}"}
            
            # Generate each distribution for this event
            futures = {}
            for dist_name, dist in distribution_configs.items():
                is_anomalous = anomaly_patterns[dist_name][event_idx]
                
                if is_anomalous:
                    anomaly_loc = random.choice(dist.anomalies)
                    sample_args = (dist, event_idx, True, anomaly_loc)
                else:
                    sample_args = (dist, event_idx, False, None)
                
                futures[dist_name] = executor.submit(generate_sample, *sample_args)
            
            # Collect results
            for dist_name, future in futures.items():
                filename, dist_name_res, anomaly_type, label, data = future.result()
                
                # Store data in appropriate detector list
                all_data[dist_name].append(data)
                
                # Store metadata
                event_metadata[f"{dist_name}_anomaly"] = anomaly_type
                event_metadata[f"{dist_name}_label"] = label
            
            metadata.append(event_metadata)

    # Save each detector's data separately as numpy arrays
    for dist_name, data_list in all_data.items():
        detector_data = np.array(data_list)
        os.makedirs(f"{config.output_dir}/numpy/{n_events}_events", exist_ok=True)
        detector_file = f"{config.output_dir}/numpy/{n_events}_events/{dist_name}.npy"
        np.save(detector_file, detector_data)
    
    # Save metadata
    labels_file = f"{config.output_dir}/labels/labels_{n_events}_events.csv"
    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(labels_file, index=False)
    
    print(f"Generated {n_events} events with {len(distribution_configs)} detectors")