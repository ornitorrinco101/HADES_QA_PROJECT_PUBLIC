import sys
import os
from timeit import default_timer as timer
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import lightning as L
import numpy as np
import pandas as pd
from pathlib import Path
import awkward as ak

root_dir = Path(__file__).parents[1] # Goes up 1 levels to Hades_Project_Tests
sys.path.append(str(root_dir))
print(root_dir)

from .config_distributions import anomaliesDefinitions as anom_def
from .config_distributions import distributionDefinitions as distr_def
from .config_distributions import artificialDataset as set_data
from .config_distributions import HADES_Artifice as set_data_HADES



class HistDL_v1(Dataset):
    def __init__(self, rootfile, n_total_per_type, anomaly_fraction, hist_bins, gen_new_data=False):
        start = timer()
        self.rootfile = rootfile
        self.n_total_per_type = n_total_per_type
        self.anomaly_fraction = anomaly_fraction
        self.hist_bins = hist_bins
        self.gen_new_data = gen_new_data

        dir_files = f"{self.rootfile}/hist_dataset"
        combined_npy = f"{dir_files}/npy/histograms_{self.n_total_per_type}_hpt.npy"
        labels_file = f"{dir_files}/labels/labels_{self.n_total_per_type}_hpt.csv"

        if self.gen_new_data or not os.path.exists(combined_npy):
            config = set_data.DatasetConfig(
                output_dir=f"{rootfile}/hist_dataset",
                n_total_per_type=self.n_total_per_type,
                anomaly_fraction=self.anomaly_fraction,
                hist_bins=self.hist_bins,
            )
            set_data.generate_dataset(config)

        # Load all histograms at once
        self.sampleIn = torch.from_numpy(np.load(combined_npy)).to(dtype=torch.float32)
        
        # Load metadata
        df = pd.read_csv(labels_file)
        self.target_tags = ['none','empty_bin','skew','secondary_peak','noisy_bin']
        self.tag_to_num = {tag: idx for idx, tag in enumerate(self.target_tags)}
        self.num_to_tag = {idx: tag for tag, idx in self.tag_to_num.items()}

        num_classes = len(self.target_tags)

        # Convert labels to one-hot encoded tensors
        self.sampleOut = torch.from_numpy(np.array([self.tag_to_num[tag] for tag in df['anomaly_type']])).to(dtype=torch.long)
        
        self.length = len(self.sampleIn)
        
        print(f"Data loaded in: {timer()-start:.2f} seconds for {self.length} samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sampleIn[idx], self.sampleOut[idx]


class HistDL_v2_test(Dataset):
    def __init__(self, rootfile, n_total_per_type, anomaly_fraction, hist_bins, gen_new_data=False,config_path="/DataLoaders/config_distributions/distribution_configs.yaml",what_dim="1D"):
        start = timer()
        self.rootfile = rootfile
        self.n_total_per_type = n_total_per_type
        self.anomaly_fraction = anomaly_fraction
        self.hist_bins = hist_bins
        self.gen_new_data = gen_new_data
        self.what_dim=what_dim

        dir_files = f"{self.rootfile}/HADES_t1_dataset" 
        numpy_dir = f"{dir_files}/numpy/{self.n_total_per_type}_events"
        labels_file = f"{dir_files}/labels/labels_{self.n_total_per_type}_events.csv"

        if self.gen_new_data or not os.path.exists(numpy_dir):
            config = set_data_HADES.DatasetConfig(
                output_dir=dir_files,
                n_total_per_type=self.n_total_per_type,
                anomaly_fraction=self.anomaly_fraction,
                hist_bins=self.hist_bins,
                config_path=config_path,
                root_dir=self.rootfile
            )
            set_data_HADES.generate_dataset(config)

        # Load each detector's data separately
        self.detector_data = {}
        if self.what_dim=="1D":
            detector_files = {
                "START_beam_mod1": f"{numpy_dir}/START_beam_mod1.npy",
                "START_beam_mod2": f"{numpy_dir}/START_beam_mod2.npy",
                "RICH_trends": f"{numpy_dir}/RICH_trends.npy",
                "ToF_multiplicity": f"{numpy_dir}/ToF_multiplicity.npy",
                "ToF_sum": f"{numpy_dir}/ToF_sum.npy",
            }
        elif self.what_dim=="2D":
            detector_files = {
                "RICH_CalsXY": f"{numpy_dir}/RICH_CalsXY.npy"
            }
        else:
            print("What dimension are those histograms now?")
        
        for detector, filepath in detector_files.items():
            if os.path.exists(filepath):
                self.detector_data[detector] = torch.from_numpy(np.load(filepath)).float()
                print(f"Loaded {detector}: {self.detector_data[detector].shape}")
        
        # Load metadata
        df = pd.read_csv(labels_file)
        if self.what_dim=="1D":
            self.target_tags = ['none','empty_bin','noisy_bin','sparse_bin','comb_artifact',
                            'uniform_noise','skew','edge_effects','secondary_peak'
                            ]
        elif self.what_dim=="2D":
            self.target_tags=['none','empty_patch','noisy_patch','sparse_patch',]
        self.tag_to_num = {tag: idx for idx, tag in enumerate(self.target_tags)}
        self.num_to_tag = {idx: tag for tag, idx in self.tag_to_num.items()}
        
        # Convert all detector labels to numerical format
        self.detector_labels = {}
        for detector in self.detector_data.keys():
            col_name = f"{detector}_anomaly"
            if col_name in df.columns:
                labels = np.array([self.tag_to_num[tag] for tag in df[col_name]])
                self.detector_labels[detector] = torch.from_numpy(labels).long()
        
        self.length = self.n_total_per_type

        print(f"Data verification:")
        print(f"Total events: {self.length}")
        print(f"Available detectors: {list(self.detector_data.keys())}")
        
        print(f"Data loaded in: {timer()-start:.2f} seconds for {self.length} events")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get data for all detectors at this index
        sampleIn = {}
        sampleOut = {}
        
        for detector, data_tensor in self.detector_data.items():
            sampleIn[detector] = data_tensor[idx]
            sampleOut[detector] = self.detector_labels[detector][idx]
        
        return sampleIn, sampleOut

    def get_detector(self, detector_name):
        """Get all data for a specific detector"""
        return self.detector_data[detector_name], self.detector_labels[detector_name]


class DataModule(L.LightningDataModule):
    """
    Loads the DataLoader we want by number:

    1: Original DataLoader
    2: Speed efficient 1 ROOT file concatenated input
    3: Speed efficient 1 ROOT file 3 tensor padded input
    4: Speed efficient many ROOT file concatenated input
    5: Speed efficient many ROOT file 3 tensor padded input (WIP!!)
    """

    def __init__(self,dataset,rootfile,n_total_per_type,anomaly_fraction,hist_bins,
                batch_size,num_workers,gen_new_data=False,what_dim="1D"):
        """
        Args:
            rootfile (string): path of the root file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()

        self.dataset=dataset
        self.rootfile=rootfile
        self.n_total_per_type=n_total_per_type
        self.anomaly_fraction=anomaly_fraction
        self.hist_bins=hist_bins
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.gen_new_data=gen_new_data
        self.what_dim=what_dim

    def prepare_data(self):
        if self.dataset=="HistDL_v1":
            self.ROOTDataset=HistDL_v1(self.rootfile,
                                   self.n_total_per_type,
                                   self.anomaly_fraction,
                                   self.hist_bins,
                                   self.gen_new_data)     
        if self.dataset=="test":
            self.ROOTDataset=HistDL_v2_test(self.rootfile,
                                   self.n_total_per_type,
                                   self.anomaly_fraction,
                                   self.hist_bins,
                                   self.gen_new_data,
                                   what_dim=self.what_dim)       
    def setup(self, stage=None):
        nb_events = len(self.ROOTDataset)
        nb_train = nb_events * 8 // 10
        nb_test = nb_events * 1 // 10
        nb_val = nb_events - nb_train - nb_test
        print("setup ", nb_events, " : ", nb_train, " | ", nb_test, " | ", nb_val)
        self.ROOTset_train, self.ROOTset_val, self.ROOTset_test = random_split(self.ROOTDataset, [nb_train, nb_val, nb_test])

    def train_dataloader(self):
        return DataLoader(dataset=self.ROOTset_train,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          )

    def val_dataloader(self):
        return DataLoader(self.ROOTset_val, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ROOTset_test, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=False
                          )