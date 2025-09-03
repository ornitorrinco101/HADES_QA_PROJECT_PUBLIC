import sys
import os

import yaml
import torch
from torch import nn
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import numpy as np
import lightning as L


import sys
import os
from pathlib import Path
import re

import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parents[2] # Goes up 2 levels to ANOMALYDETECTIONHADES
sys.path.append(str(root_dir))
print(root_dir)

from Models import CNN_models as CNN
from DataLoaders import data_loaders as DL

from timeit import default_timer as timer

device='cuda:1' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
#load data to Data
torch.cuda.manual_seed(42)
torch.manual_seed(42)

start=timer()

version=2
n_total_per_type=10000
bins=50
batch_size=10
gen_new_data=False
anomaly_fraction=0.5
what_dim="1D"


Data=DL.DataModule(dataset="test",
            rootfile=root_dir,
            n_total_per_type=n_total_per_type,
            anomaly_fraction=anomaly_fraction,
            hist_bins=bins,
            batch_size=batch_size,
            num_workers=12,
            gen_new_data=gen_new_data,
            what_dim=what_dim
            )

Data.prepare_data()
Data.setup()

lr=1e-3

# Model_CNN = CNN.HADES_VAE_Model(lr=lr,version=version,
#                         hidden_units=32,
#                         kernel_size_convolution=2,
#                         kernel_size_maxpool=2,
#                         stride=1,
#                         padding=1,
#                         latent_dim=32,
#                         what_dim=what_dim).to(device)
log_dir=root_dir/f"logs/VAE/Model-{version}/{n_total_per_type}_hpt/"
Model_CNN=CNN.HADES_VAE_OTHER(
    lr=lr,
    version=2,
    hidden_units=8,
    kernel_size_convolution=3,
    kernel_size_maxpool=2,
    stride=1,
    padding=1,
    # anomaly_threshold=0.1,
    # log_dir=log_dir
)

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

logger = CSVLogger(log_dir, name=f"{anomaly_fraction:.1f}_poa")

epochs=30


trainer = L.Trainer(default_root_dir=root_dir,
                        callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss")],
                        # accelerator="gpu",
                        # devices=[0],
                        # devices=2,
                        # strategy='ddp',
                        # precision='16-mixed',
                        max_epochs=epochs,
                        # gradient_clip_val=5,
                        #accumulate_grad_batches=4,
                        #precision="16-mixed",
                        logger=logger,
                        enable_progress_bar=True,
                        check_val_every_n_epoch=1
                        # resume_from_checkpoint=Path
                        )

trainer.fit(Model_CNN,
                Data.train_dataloader(),
                Data.val_dataloader(),
                # ckpt_path=Path
                )

trainer.test(Model_CNN,Data.test_dataloader())


end=timer()
print(f"\nIt all took a total of {(end-start)/60:.2f} minutes, or {(end-start)/3600:.2f} hours.\n")
