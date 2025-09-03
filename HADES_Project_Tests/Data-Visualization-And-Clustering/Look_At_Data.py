import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_loss(root_dir,model,version_model,events,anomaly_fraction,log_number,c,l):
    filepath=f"{root_dir}/logs/{model}/Model-{version_model}/{events}_hpt/{anomaly_fraction}_poa/version_{log_number}/metrics.csv"

    metrics=pd.read_csv(filepath,delimiter=",")

    t_loss=metrics["train_loss"]
    v_loss=metrics["val_loss"]

    cond1=(np.isnan(t_loss)==False)
    cond2=(np.isnan(v_loss)==False)

    epoch=metrics["epoch"]

    plt.plot(epoch[cond1],t_loss[cond1],color=c[0],label=l[0])
    plt.plot(epoch[cond2],v_loss[cond2],color=c[1],label=l[1])

def confuse_matrix(root_dir, model, version_model, events, anomaly_fraction, log_number, nsamples):
    if version_model==2:
        keys = ["START_beam_mod1", "START_beam_mod2", "RICH_trends", "ToF_multiplicity", "ToF_sum"]
    elif version_model==3:
        keys=["RICH_CalsXY"]
    filepath = f"{root_dir}/logs/{model}/Model-{version_model}/{events}_hpt/{anomaly_fraction}_poa/version_{log_number}/"

    # Read data with error handling
    try:
        output = pd.read_csv(filepath + "outputs_max.csv", delimiter=",")
        target = pd.read_csv(filepath + "targets.csv", delimiter=",")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Ensure we don't exceed available samples
    # nsamples = min(nsamples, len(output), len(target))
    
    if version_model==2:
        class_names = ['none', 'empty_bin', 'noisy_bin', 'sparse_bin', 'comb_artifact', 
                    'uniform_noise', 'skew', 'edge_effects', 'secondary_peak']
    elif version_model==3:
        class_names=['none','empty_patch','noisy_patch','sparse_patch']

    # Create subplots
    if version_model==2:
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))  # Slightly larger for better readability
        axes = axes.ravel()
    elif version_model==3:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))  # Slightly larger for better readability
        axes=[axes]

    # Detector names
    detector_names = keys

    for i, key in enumerate(keys):
        # Check if key exists in both dataframes
        if key not in output.columns or key not in target.columns:
            print(f"Warning: Key '{key}' not found in data. Skipping.")
            axes[i].set_visible(False)
            continue
        
        # Extract data for this detector
        y_true = target[key][0:nsamples].astype(int)
        y_pred = output[key][0:nsamples].astype(int)
        
        # Check if we have any data
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"Warning: No data for detector '{key}'. Skipping.")
            axes[i].set_visible(False)
            continue
        
        # Calculate confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        except Exception as e:
            print(f"Error calculating confusion matrix for {key}: {e}")
            axes[i].set_visible(False)
            continue
        
        # Normalize the confusion matrix safely
        row_sums = cm.sum(axis=1)
        cm_normalized = np.zeros_like(cm, dtype=float)
        for j in range(cm.shape[0]):
            if row_sums[j] > 0:
                cm_normalized[j, :] = cm[j, :].astype(float) / row_sums[j]
        
        # Plot confusion matrix
        try:
            sns.heatmap(cm_normalized, 
                        annot=True, 
                        fmt='.2f', 
                        cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=axes[i],
                        cbar_kws={'shrink': 0.8},
                        vmin=0, vmax=1)  # Fixed scale for better comparison
            
            if version==2:
                axes[i].set_title(f'{detector_names[i]}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted Label', fontsize=12)
            axes[i].set_ylabel('True Label', fontsize=12)
            axes[i].set_title(f'{key}',fontsize=15)
            
            # Rotate x labels for better readability
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[i].get_yticklabels(), rotation=0)
            
        except Exception as e:
            print(f"Error plotting for {key}: {e}")
            axes[i].set_visible(False)

    # Remove the empty subplot (6th one)
    if version==2:
        fig.delaxes(axes[5])

    plt.tight_layout()
    
    # Add overall title
    # fig.suptitle(f'Confusion Matrices - Model {model} v{version_model}\n
                # f'Events: {events}, Anomaly: {anomaly_fraction}, Samples: {nsamples}', 
                # fontsize=16, y=1.02)
    
    # Save the figure
    save_path = f"{filepath}confusion_matrix.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def combined_confuse_matrix(root_dir, model, events, anomaly_fraction, log_number_v2, log_number_v3, nsamples):
    # Define parameters for both versions
    versions = [2, 3]
    log_numbers = [log_number_v2, log_number_v3]
    
    # Create the figure with the original 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.ravel()
    
    # Process version 2 first (will use first 5 subplots)
    version = 2
    log_number = log_number_v2
    keys = ["START_beam_mod1", "START_beam_mod2", "RICH_trends", "ToF_multiplicity", "ToF_sum"]
    class_names = ['none', 'empty_bin', 'noisy_bin', 'sparse_bin', 'comb_artifact', 
                  'uniform_noise', 'skew', 'edge_effects', 'secondary_peak']
    
    filepath = f"{root_dir}/logs/{model}/Model-{version}/{events}_hpt/{anomaly_fraction}_poa/version_{log_number}/"
    
    # Read data for version 2
    try:
        output_v2 = pd.read_csv(filepath + "outputs_max.csv", delimiter=",")
        target_v2 = pd.read_csv(filepath + "targets.csv", delimiter=",")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    # Plot version 2 detectors in first 5 subplots
    for i, key in enumerate(keys):
        ax = axes[i]
        
        # Check if key exists in both dataframes
        if key not in output_v2.columns or key not in target_v2.columns:
            print(f"Warning: Key '{key}' not found in data. Skipping.")
            ax.set_visible(False)
            continue
        
        # Extract data for this detector
        y_true = target_v2[key][0:nsamples].astype(int)
        y_pred = output_v2[key][0:nsamples].astype(int)
        
        # Check if we have any data
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"Warning: No data for detector '{key}'. Skipping.")
            ax.set_visible(False)
            continue
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        except Exception as e:
            print(f"Error calculating confusion matrix for {key}: {e}")
            ax.set_visible(False)
            continue
        
        # Find which rows and columns have non-zero entries
        non_zero_rows = np.where(cm.sum(axis=1) > 0)[0]
        non_zero_cols = np.where(cm.sum(axis=0) > 0)[0]
        
        # If no non-zero entries, skip this plot
        if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
            print(f"Warning: No predictions for detector '{key}'. Skipping.")
            ax.set_visible(False)
            continue
        
        # Filter the confusion matrix and class names
        cm_filtered = cm[non_zero_rows][:, non_zero_cols]
        filtered_class_names = [class_names[i] for i in non_zero_rows]
        
        # Normalize the confusion matrix safely
        row_sums = cm_filtered.sum(axis=1)
        cm_normalized = np.zeros_like(cm_filtered, dtype=float)
        for j in range(cm_filtered.shape[0]):
            if row_sums[j] > 0:
                cm_normalized[j, :] = cm_filtered[j, :].astype(float) / row_sums[j]
        
        # Plot confusion matrix
        try:
            sns.heatmap(cm_normalized, 
                        annot=True, 
                        fmt='.2f', 
                        cmap='Blues',
                        xticklabels=[class_names[i] for i in non_zero_cols],
                        yticklabels=filtered_class_names,
                        ax=ax,
                        cbar_kws={'shrink': 0.8},
                        vmin=0, vmax=1)
            
            # Add accuracy to the title
            ax.set_title(f'{key}\n Acc: {accuracy:.2f}', fontsize=15)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
            
        except Exception as e:
            print(f"Error plotting for {key}: {e}")
            ax.set_visible(False)
    
    # Now process version 3 and plot in the 6th subplot (axes[5])
    version = 3
    log_number = log_number_v3
    keys = ["RICH_CalsXY"]
    class_names = ['none', 'empty_patch', 'noisy_patch', 'sparse_patch']
    
    filepath = f"{root_dir}/logs/{model}/Model-{version}/{events}_hpt/{anomaly_fraction}_poa/version_{log_number}/"
    
    # Read data for version 3
    try:
        output_v3 = pd.read_csv(filepath + "outputs_max.csv", delimiter=",")
        target_v3 = pd.read_csv(filepath + "targets.csv", delimiter=",")
    except FileNotFoundError as e:
        print(f"Error: File not found for version 3 - {e}")
        axes[5].set_visible(False)
    except Exception as e:
        print(f"Error reading files for version 3: {e}")
        axes[5].set_visible(False)
    else:
        # Plot version 3 in the 6th subplot
        ax = axes[5]
        key = keys[0]
        
        if key in output_v3.columns and key in target_v3.columns:
            # Extract data for this detector
            y_true = target_v3[key][0:nsamples].astype(int)
            y_pred = output_v3[key][0:nsamples].astype(int)
            
            if len(y_true) > 0 and len(y_pred) > 0:
                # Calculate accuracy
                accuracy = accuracy_score(y_true, y_pred)
                
                # Calculate confusion matrix
                try:
                    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
                except Exception as e:
                    print(f"Error calculating confusion matrix for {key}: {e}")
                    ax.set_visible(False)
                else:
                    # Find which rows and columns have non-zero entries
                    non_zero_rows = np.where(cm.sum(axis=1) > 0)[0]
                    non_zero_cols = np.where(cm.sum(axis=0) > 0)[0]
                    
                    # If no non-zero entries, skip this plot
                    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
                        print(f"Warning: No predictions for detector '{key}'. Skipping.")
                        ax.set_visible(False)
                    else:
                        # Filter the confusion matrix and class names
                        cm_filtered = cm[non_zero_rows][:, non_zero_cols]
                        filtered_class_names = [class_names[i] for i in non_zero_rows]
                        
                        # Normalize the confusion matrix safely
                        row_sums = cm_filtered.sum(axis=1)
                        cm_normalized = np.zeros_like(cm_filtered, dtype=float)
                        for j in range(cm_filtered.shape[0]):
                            if row_sums[j] > 0:
                                cm_normalized[j, :] = cm_filtered[j, :].astype(float) / row_sums[j]
                        
                        # Plot confusion matrix
                        try:
                            sns.heatmap(cm_normalized, 
                                        annot=True, 
                                        fmt='.2f', 
                                        cmap='Blues',
                                        xticklabels=[class_names[i] for i in non_zero_cols],
                                        yticklabels=filtered_class_names,
                                        ax=ax,
                                        cbar_kws={'shrink': 0.8},
                                        vmin=0, vmax=1)
                            
                            # Add accuracy to the title
                            ax.set_title(f'{key}\n Acc: {accuracy:.2f}', fontsize=15)
                            ax.set_xlabel('Predicted Label', fontsize=12)
                            ax.set_ylabel('True Label', fontsize=12)
                            
                            # Rotate x labels for better readability
                            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                            plt.setp(ax.get_yticklabels(), rotation=0)
                            
                        except Exception as e:
                            print(f"Error plotting for {key}: {e}")
                            ax.set_visible(False)
            else:
                print(f"Warning: No data for detector '{key}'. Skipping.")
                ax.set_visible(False)
        else:
            print(f"Warning: Key '{key}' not found in version 3 data. Skipping.")
            ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = f"{root_dir}/logs/{model}/combined_confusion_matrix_v2_v3.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Combined confusion matrix saved to: {save_path}")
    
    plt.show()

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KernelDensity
from sklearn.svm import OneClassSVM
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.stats import genpareto
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# optional: HDBSCAN
try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

def _central_cluster(labels, X):
    """Pick cluster id (excl. -1) whose centroid is Mahalanobis-closest to global mean."""
    uniq = [c for c in np.unique(labels) if c != -1]
    if len(uniq) == 0:
        return None
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    try:
        VI = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        VI = np.eye(X.shape[1])
    best_c, best_d = None, np.inf
    for c in uniq:
        m = (labels == c)
        if not np.any(m): 
            continue
        centroid = X[m].mean(axis=0)
        try:
            d = mahalanobis(mu, centroid, VI)
        except Exception:
            d = np.linalg.norm(mu - centroid)
        if d < best_d:
            best_d, best_c = d, c
    return best_c

def _evt_threshold(scores, tail_frac=0.05, q=0.99):
    """Peaks-over-threshold with Generalized Pareto on upper tail of scores."""
    scores = np.asarray(scores)
    u = np.quantile(scores, 1.0 - tail_frac)
    tail = scores[scores >= u] - u
    if len(tail) < 10:
        # fallback: plain high quantile if too few tail points
        return np.quantile(scores, q)
    # Fit GPD (shape c, loc=0, scale)
    c, loc, scale = genpareto.fit(tail, floc=0.0)
    # Quantile of tail, then add back u
    tail_q = genpareto.ppf(q, c, loc=0.0, scale=scale)
    return float(u + max(0.0, tail_q))

# Figure 2
# ==============================
def per_detector_tsne_unsupervised(
    root_dir, Model_version, events, poa, version,
    anomaly_names=None,
    detectors=("START_beam_mod1","START_beam_mod2","RICH_trends","ToF_multiplicity","ToF_sum"),
    method="hdbscan",
    max_samples=2000,
    perplexity=30,
    random_state=42,
    standardize=False,
    pca_components=None,
    k_max=6,
    contamination=0.05,
    min_cluster_size=25,
    min_samples=10,
    eps=0.8,
    n_neighbors=20,
    nu=0.05,
    bandwidth=None,
    quantile_override=None,
    use_evt=False,
    evt_tail_frac=0.05
    ):
    """
    Unsupervised per-detector anomaly detection with multiple methods.
    Includes detailed confusion matrices for each detector.
    """

    # canonical anomaly names + consistent color map
    if anomaly_names is None:
        anomaly_names = _default_anomaly_names()
    else:
        anomaly_names = {k: _canon_label(v, _default_anomaly_names()) for k, v in anomaly_names.items()}
    color_map = _fixed_color_map(anomaly_names)
    legend_order = [anomaly_names[k] for k in sorted(anomaly_names.keys())]

    base = os.path.join(root_dir, f"logs/VAE/Model-{Model_version}",
                        f"{events}_hpt", f"{poa}_poa", f"version_{version}")
    out_dir = os.path.join(base, f"TSNE_{method.upper()}_PerDetector_Unsupervised")
    os.makedirs(out_dir, exist_ok=True)

    cm_dir = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    all_results = []
    detector_metrics = []
    all_detailed_cms = []

    for det in detectors:
        print(f"\nProcessing detector: {det}")

        # ----- load features + labels -----
        try:
            X = np.load(os.path.join(base, "latent_features", f"{det}_latent_features.npy"))
            y_path = os.path.join(base, "latent_features", f"{det}_labels.csv")
            y = pd.read_csv(y_path)["label"].values if os.path.exists(y_path) else None
            print(f"Loaded {len(X)} samples for {det}")
        except FileNotFoundError:
            print(f"[{det}] missing latent_features, skipping.")
            continue

        n = len(X)
        if y is not None:
            n = min(n, len(y))
        if n > max_samples:
            idx = np.random.choice(n, max_samples, replace=False)
            X = X[idx]
            if y is not None:
                y = y[idx]
            n = len(X)
        elif y is not None:
            X, y = X[:n], y[:n]

        # ----- preprocess -----
        X_in = X.astype(np.float64)
        if standardize:
            X_in = StandardScaler().fit_transform(X_in)

        # Optional PCA
        if pca_components is not None:
            if isinstance(pca_components, float) and 0 < pca_components <= 1.0:
                pca = PCA(n_components=pca_components, svd_solver="full", random_state=random_state)
            elif isinstance(pca_components, int) and pca_components >= 1:
                pca = PCA(n_components=pca_components, random_state=random_state)
            else:
                pca = None
            if pca is not None:
                X_in = pca.fit_transform(X_in)

        # ----- method-specific predictions -----
        pred_bin = None
        score = None
        cluster_labels = None
        info = ""

        if method == "hdbscan":
            if not _HAS_HDBSCAN:
                raise ImportError("hdbscan is not installed. pip install hdbscan")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=min_samples, metric="euclidean")
            clusterer.fit(X_in)
            cluster_labels = clusterer.labels_
            normal_c = _central_cluster(cluster_labels, X_in)
            pred_bin = (cluster_labels == -1)
            if normal_c is not None:
                pred_bin = np.where(cluster_labels == normal_c, 0, 1)
            pred_bin = pred_bin.astype(int)

            if hasattr(clusterer, "outlier_scores_"):
                score = clusterer.outlier_scores_
            else:
                if normal_c is not None:
                    mu_norm = X_in[cluster_labels == normal_c].mean(axis=0)
                    score = np.linalg.norm(X_in - mu_norm, axis=1)

            if quantile_override is not None or use_evt:
                if score is None:
                    score = pred_bin.astype(float)
                if use_evt:
                    thr = _evt_threshold(score, tail_frac=evt_tail_frac, q=0.99)
                    info = f"EVT thr={thr:.3g}"
                else:
                    thr = np.quantile(score, quantile_override)
                    info = f"q={quantile_override:.2f}"
                pred_bin = (score > thr).astype(int)

        elif method == "dbscan":
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_in)
            cluster_labels = db.labels_
            normal_c = _central_cluster(cluster_labels, X_in)
            pred_bin = (cluster_labels == -1)
            if normal_c is not None:
                pred_bin = np.where(cluster_labels == normal_c, 0, 1)
            pred_bin = pred_bin.astype(int)
            info = f"eps={eps}, min_samples={min_samples}"

        elif method == "spectral":
            n_c = min(5, k_max)
            sc = SpectralClustering(n_clusters=n_c, assign_labels="kmeans",
                                    random_state=random_state, affinity="nearest_neighbors")
            cluster_labels = sc.fit_predict(X_in)
            normal_c = _central_cluster(cluster_labels, X_in)
            pred_bin = np.where(cluster_labels == normal_c, 0, 1).astype(int)
            info = f"n_clusters={n_c}"

        elif method == "bayes_gmm":
            bgmm = BayesianGaussianMixture(
                n_components=k_max, covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                n_init=3, random_state=random_state
            ).fit(X_in)
            resp = bgmm.predict_proba(X_in)
            comps = resp.argmax(1)
            means, covs = bgmm.means_, bgmm.covariances_
            mu = X_in.mean(axis=0)
            dists = []
            for m, c in zip(means, covs):
                try:
                    VI = np.linalg.pinv(c)
                    d = mahalanobis(mu, m, VI)
                except Exception:
                    d = np.linalg.norm(mu - m)
                dists.append(d)
            normal_c = int(np.argmin(dists))
            pred_bin = (comps != normal_c).astype(int)
            score = 1.0 - resp[:, normal_c]
            if quantile_override is not None or use_evt:
                thr = (_evt_threshold(score, evt_tail_frac, 0.99) if use_evt
                       else np.quantile(score, quantile_override))
                pred_bin = (score > thr).astype(int)
                info = ("EVT" if use_evt else f"q={quantile_override:.2f}") + f", k_max={k_max}"
            else:
                info = f"k_max={k_max}"

        elif method == "isoforest":
            iso = IsolationForest(n_estimators=300, contamination="auto", random_state=random_state)
            iso.fit(X_in)
            dfun = iso.decision_function(X_in)
            score = -dfun
            if quantile_override is not None:
                thr = np.quantile(score, quantile_override)
            elif use_evt:
                thr = _evt_threshold(score, tail_frac=evt_tail_frac, q=0.99)
            else:
                thr = np.quantile(score, 1.0 - contamination)
            pred_bin = (score > thr).astype(int)
            info = f"cont={contamination}"

        elif method == "lof":
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto", novelty=False)
            _ = lof.fit_predict(X_in)
            score = -lof.negative_outlier_factor_
            if quantile_override is not None:
                thr = np.quantile(score, quantile_override)
            elif use_evt:
                thr = _evt_threshold(score, tail_frac=evt_tail_frac, q=0.99)
            else:
                thr = np.quantile(score, 1.0 - contamination)
            pred_bin = (score > thr).astype(int)
            info = f"n_neighbors={n_neighbors}"

        elif method == "ocsvm":
            oc = OneClassSVM(kernel="rbf", gamma="scale", nu=max(1e-3, min(0.5, nu)))
            oc.fit(X_in)
            dfun = oc.decision_function(X_in)
            score = -dfun
            if quantile_override is not None:
                thr = np.quantile(score, quantile_override)
            elif use_evt:
                thr = _evt_threshold(score, tail_frac=evt_tail_frac, q=0.99)
            else:
                thr = np.quantile(score, 1.0 - contamination)
            pred_bin = (score > thr).astype(int)
            info = f"nu={nu}"

        elif method == "kde":
            if bandwidth is None:
                std = np.mean(X_in.std(axis=0))
                d = X_in.shape[1]
                bandwidth = std * (X_in.shape[0] ** (-1.0 / (d + 4)))
                bandwidth = max(bandwidth, 1e-3)
            kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(X_in)
            log_dens = kde.score_samples(X_in)
            score = -log_dens
            if quantile_override is not None:
                thr = np.quantile(score, quantile_override)
            elif use_evt:
                thr = _evt_threshold(score, tail_frac=evt_tail_frac, q=0.99)
            else:
                thr = np.quantile(score, 1.0 - contamination)
            pred_bin = (score > thr).astype(int)
            info = f"bw={bandwidth:.3g}"

        elif method == "pca_mahal":
            lw = LedoitWolf().fit(X_in)
            mu = lw.location_
            VI = lw.precision_
            diffs = X_in - mu
            score = np.einsum("ij,jk,ik->i", diffs, VI, diffs)
            if quantile_override is not None:
                thr = np.quantile(score, quantile_override)
            elif use_evt:
                thr = _evt_threshold(score, tail_frac=evt_tail_frac, q=0.99)
            else:
                thr = np.quantile(score, 1.0 - contamination)
            pred_bin = (score > thr).astype(int)
            info = "LedoitWolf Mahalanobis"

        else:
            raise ValueError(f"Unknown method '{method}'")

        # ----- metrics + confusion matrices if labels available -----
        if y is not None and len(y) == n:
            y_canon = _canon_array(y, anomaly_names)
            true_binary = np.where(y_canon == "no anomaly", 0, 1)
            accuracy = accuracy_score(true_binary, pred_bin)
            precision = precision_score(true_binary, pred_bin, zero_division=0)
            recall = recall_score(true_binary, pred_bin, zero_division=0)
            f1 = f1_score(true_binary, pred_bin, zero_division=0)

            info += f", Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}"

            detector_metrics.append({
                "detector": det,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_samples": n,
                "n_normal": np.sum(true_binary == 0),
                "n_anomaly": np.sum(true_binary == 1)
            })

            # Create detailed confusion matrices (your existing helpers)
            cm, detailed_cm, present_labels = _create_detector_confusion_matrix(
                y, pred_bin, anomaly_names, cm_dir, method, det
            )
            all_detailed_cms.append({
                "detector": det,
                "confusion_matrix": cm,
                "detailed_matrix": detailed_cm,
                "present_labels": present_labels,
                "accuracy": accuracy
            })

        # ----- t-SNE visualization -----
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        Z = tsne.fit_transform(X_in)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: TRUE labels in canonical order with fixed colors
        if y is not None and len(y) == n:
            y_canon = _canon_array(y, anomaly_names)
            for name in legend_order:
                m = (y_canon == name)
                if np.any(m):
                    axes[0].scatter(Z[m, 0], Z[m, 1],
                                    c=color_map[name], s=10, alpha=0.75, label=name)
            axes[0].legend(title="True Anomaly Type", bbox_to_anchor=(1.02, 1), loc="upper left")
            axes[0].set_title(f"{det} - True Anomaly Types")
        else:
            axes[0].scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7)
            axes[0].set_title(f"{det} - t-SNE (unlabeled)")

        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")
        axes[0].grid(True, alpha=0.3)

        # Right: predicted binary
        for lab, col, val in [("Normal (Pred)", "green", 0), ("Anomaly (Pred)", "red", 1)]:
            m = (pred_bin == val)
            if np.any(m):
                axes[1].scatter(Z[m, 0], Z[m, 1], c=col, s=10, alpha=0.75, label=lab)

        title = f"{det} - {method.upper()} Prediction"
        if y is not None:
            title += f"\nAcc={accuracy:.3f}, F1={f1:.3f}"
        axes[1].set_title(title)
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(title="Predicted")

        plt.tight_layout()
        tsne_path = os.path.join(out_dir, f"{det}_tsne_{method}.png")
        plt.savefig(tsne_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"TSNE plot saved to: {tsne_path}")

        # ----- collect results -----
        df = {
            "detector": [det] * n,
            "pred_binary": pred_bin,
            "tsne_x": Z[:, 0],
            "tsne_y": Z[:, 1],
            "method": [method] * n
        }

        if y is not None and len(y) == n:
            df["true_label"] = y
            true_binary = np.where(_canon_array(y, anomaly_names) == "no anomaly", 0, 1)
            df["correct"] = (pred_bin == true_binary).astype(int)

        if score is not None:
            df["score"] = score
        if cluster_labels is not None:
            df["cluster"] = cluster_labels

        all_results.append(pd.DataFrame(df))

        print(f"[{det}] {method} | {info}")

    # ----- summary confusion matrix -----
    if all_detailed_cms:
        _create_summary_confusion_matrix(all_detailed_cms, anomaly_names, cm_dir, method)

    if not all_results:
        print("No detectors processed.")
        return None

    results = pd.concat(all_results, ignore_index=True)

    if detector_metrics:
        metrics_df = pd.DataFrame(detector_metrics)
        metrics_path = os.path.join(out_dir, f"detector_metrics_{method}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")

    results_path = os.path.join(out_dir, f"per_detector_{method}_unsupervised.csv")
    results.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    return results

def _create_detector_confusion_matrix(true_labels, pred_labels, anomaly_names, out_dir, method, detector_name):
    """
    Create detailed confusion matrix for a single detector showing all anomaly types
    """
    # Convert to binary for the confusion matrix (0=normal, 1=anomaly)
    true_binary = np.where(true_labels == 0, 0, 1)
    
    # Create confusion matrix
    cm = confusion_matrix(true_binary, pred_labels, labels=[0, 1])
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                ax=ax1, cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)
    ax1.set_title(f'{detector_name} - Normalized Confusion Matrix\n{method.upper()}')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot 2: Detailed breakdown by anomaly type - FILTER EMPTY LABELS
    # Get unique labels that actually appear in the data
    unique_true = np.unique(true_labels)
    
    # Filter out labels that have no instances
    present_labels = []
    label_counts = []
    for lbl in unique_true:
        count = np.sum(true_labels == lbl)
        if count > 0:
            present_labels.append(lbl)
            label_counts.append(count)
    
    # Create matrix: rows = present true anomaly types, columns = predicted (0, 1)
    detailed_cm = np.zeros((len(present_labels), 2))
    for i, true_label in enumerate(present_labels):
        mask = (true_labels == true_label)
        detailed_cm[i, 0] = np.sum((pred_labels[mask] == 0))
        detailed_cm[i, 1] = np.sum((pred_labels[mask] == 1))
    
    # Normalize by row
    row_sums = detailed_cm.sum(axis=1)
    detailed_cm_norm = np.zeros_like(detailed_cm, dtype=float)
    for i in range(len(present_labels)):
        if row_sums[i] > 0:
            detailed_cm_norm[i, :] = detailed_cm[i, :] / row_sums[i]
    
    # Create labels for y-axis
    y_labels = [anomaly_names.get(lbl, f'Label {lbl}') for lbl in present_labels]
    
    sns.heatmap(detailed_cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['Pred Normal', 'Pred Anomaly'],
                yticklabels=y_labels,
                ax=ax2, cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)
    ax2.set_title(f'{detector_name} - Detailed Anomaly Type Breakdown\n{method.upper()}')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Anomaly Type')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"{detector_name}_confusion_matrix_{method}.png")
    plt.savefig(cm_path, dpi=220, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Return both the actual labels present and the confusion matrix
    return cm, detailed_cm, present_labels

def _create_summary_confusion_matrix(all_detailed_cms, anomaly_names, cm_dir, method):
    """
    Create a summary confusion matrix showing all detectors in one plot
    """
    n_detectors = len(all_detailed_cms)
    if n_detectors == 0:
        return
    
    # Get all unique anomaly labels that actually appear in any detector
    all_present_labels = set()
    for det_data in all_detailed_cms:
        all_present_labels.update(det_data["present_labels"])
    
    all_present_labels = sorted(all_present_labels)
    
    n_cols = min(3, n_detectors)
    n_rows = (n_detectors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    for i, det_data in enumerate(all_detailed_cms):
        det = det_data["detector"]
        detailed_cm = det_data["detailed_matrix"]
        present_labels = det_data["present_labels"]
        accuracy = det_data["accuracy"]
        
        # Create a mapping from label to row index in this detector
        label_to_index = {lbl: idx for idx, lbl in enumerate(present_labels)}
        
        # Only include labels that are present in THIS detector
        detector_present_labels = [lbl for lbl in all_present_labels if lbl in present_labels]
        
        # Create a matrix with only the labels present in this detector
        filtered_detailed_cm = np.zeros((len(detector_present_labels), 2))
        filtered_detailed_cm_norm = np.zeros((len(detector_present_labels), 2))
        
        # Fill the matrix with data
        for j, lbl in enumerate(detector_present_labels):
            idx = label_to_index[lbl]
            filtered_detailed_cm[j, :] = detailed_cm[idx, :]
            
            # Normalize this row
            row_sum = detailed_cm[idx, :].sum()
            if row_sum > 0:
                filtered_detailed_cm_norm[j, :] = detailed_cm[idx, :] / row_sum
        
        # Create labels for y-axis
        y_labels = [anomaly_names.get(lbl, f'Label {lbl}') for lbl in detector_present_labels]
        
        # Create a DataFrame for seaborn heatmap
        df = pd.DataFrame(filtered_detailed_cm_norm, 
                         index=y_labels, 
                         columns=['Normal', 'Anomaly'])
        
        sns.heatmap(df, annot=True, fmt='.3f', cmap='Blues',
                    ax=axes[i], cbar_kws={'shrink': 0.7}, vmin=0, vmax=1)
        
        axes[i].set_title(f'{det}\nAcc={accuracy:.2f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True Type')
        plt.setp(axes[i].get_yticklabels(), rotation=0)
    
    # Hide empty subplots
    for i in range(len(all_detailed_cms), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    summary_path = os.path.join(cm_dir, f"summary_confusion_matrix_{method}.png")
    plt.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Summary confusion matrix saved to: {summary_path}")

def _default_anomaly_names():
    # canonical labels with spaces, in the exact order you want in the legend
    return {
        0: "no anomaly",
        1: "empty bin",
        2: "noisy bin",
        3: "sparse bin",
        4: "comb artifact",
        5: "uniform noise",
        6: "skewed distribution",
        7: "edge effects",
        8: "secondary peak",
    }

# Tableau-10 palette (widely used and color-blind-friendly)
_TABLEAU10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def _fixed_color_map(anomaly_names_dict):
    # Build a name->color map in dictionary order
    order = [anomaly_names_dict[k] for k in sorted(anomaly_names_dict.keys())]
    cmap = {}
    for i, name in enumerate(order):
        cmap[name] = _TABLEAU10[i % len(_TABLEAU10)]
    return cmap

# Map many possible string variants to canonical labels with spaces
_VARIANTS = {
    "none": "no anomaly",
    "no_anomaly": "no anomaly",
    "no anomaly": "no anomaly",
    "empty_bin": "empty bin",
    "empty bin": "empty bin",
    "noisy_bin": "noisy bin",
    "noisy bin": "noisy bin",
    "sparse_bin": "sparse bin",
    "sparse bin": "sparse bin",
    "comb_artifact": "comb artifact",
    "comb artifact": "comb artifact",
    "uniform_noise": "uniform noise",
    "uniform noise": "uniform noise",
    "skewed_distribution": "skewed distribution",
    "skewed distribution": "skewed distribution",
    "edge_effects": "edge effects",
    "edge effects": "edge effects",
    "secondary_peak": "secondary peak",
    "secondary peak": "secondary peak",
}

def _canon_label(x, anomaly_names_dict):
    if isinstance(x, (np.integer, int)):
        return anomaly_names_dict.get(int(x), str(x))
    if isinstance(x, str):
        key = x.strip().lower().replace("-", " ").replace("_", " ")
        return _VARIANTS.get(key, key)
    return str(x)

def _canon_array(y, anomaly_names_dict):
    return np.array([_canon_label(v, anomaly_names_dict) for v in y])