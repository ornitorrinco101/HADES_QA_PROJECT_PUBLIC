
# ML for Anomaly Detection at HADES

This repository features initial tests evaluating the performance of supervised/unsupervised models on simulated data with injected anomalies.



## Authors

- [@ornitorrinco101](https://www.github.com/ornitorrinco101)

- [@KladovValentin]((https://github.com/KladovValentin)) (initial code)
## Setup

Project requires python libraries within req.txt. 

To implement:

```pip install -r req.txt```

`umap-learn` can be removed for current evaluations.
## Documentation

### Data Loading: `DataLoaders/config_distributions`

 - `distribution_configs.yaml`:
    Sets distribution and anomaly parameters.

 - `anomaliesDefinitions.py`:
    Defines anomalies in QA plots.
 
 - `distributionDefinitions.py`:
    Defines distributions for QA plots.

- `HADES_Artifice.py`:
    Generates QA plots which are stored in `HADES_t1_dataset`.

### ML Models: `Models/CNN_models.py`

#### Supervised Models:

- `HADES_5S_1D()`:
    5-system model for 1D histogram plots.
    - START mod0 (layer1), START mod1 (layer 2)
    - RICH Trends
    - ToF Multiplicity
    - Tof Sum
- `HADES_1S_2D()`:
    1-system model for 2D image data.
    - RICH CALS XY

#### Unsupervised Model:

- `HADES_VAE_OTHER()`:
    Variational Autoencoder for `5S_1D` data.

### Training: `Trainers`

- `CNN`: Supervised models.

- `VAE`: Unsupervised models.



### Logs: `logs`

- `CNN`: Supervised model logs.
    - `Model-1`: 
        Evaluation with original [@KladovValentin]((https://github.com/KladovValentin/anomalyDetectionHades)) code. 
    - `Model-2`: 
        Evaluation with `HADES_5S_1D()`.
    - `Model-3`: 
        Evaluation with `HADES_1S_2D()`.

    Hpt: Total number of QA events\
    Poa: Probability of anomaly (0-1)

- `VAE`: Unsupervised model logs.
    - `Model-2`: 
        Evaluation with `HADES_VAE_OTHER()`.
    `latent_features`: saves latent space parameters per QA plot.

### Evaluation: `Data-Visualization-And-Clustering`

- Confusion matrices: `Look_At_Data.py`\
    Create confusion matrices for supervised models: `confuse_matrix()`\
    Create loss plots for all models: `plot_loss()`

- Clustering algorithms: `Cluster_and_Confuse.py`\
    Clusters input data and outputs confusion matrix plots in `logs` or `HADES_t1_dataset`.\
    Model: HDBSCAN\
    Parameters: `min_cluster_size=2`, `min_samples=2`
    - `per_detector_tsne_dataloader()`:
        Clustering on input QA plots.
    - `per_detector_tsne_unsupervised()`:
        Clustering on `latent_features`.







