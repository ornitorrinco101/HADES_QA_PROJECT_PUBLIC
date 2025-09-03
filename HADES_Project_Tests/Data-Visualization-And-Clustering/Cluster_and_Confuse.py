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
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
from scipy.spatial.distance import mahalanobis
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_anomaly_detection(true_labels, pred_labels, anomaly_classes):
    """
    true_labels: array of true labels (categorical anomaly types + 'no anomaly')
    pred_labels: array of predicted labels from HDBSCAN (0=normal, 1=anomaly)
    anomaly_classes: list of label names that count as anomaly
    """

    # convert to binary (0 = normal, 1 = anomaly)
    y_true = [0 if l == "no anomaly" else 1 for l in true_labels]
    y_pred = pred_labels  # already 0/1

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", cm)
    
    return acc, prec, rec, f1, cm

def per_detector_tsne_unsupervised(
    root_dir, Model_version, events, poa, version,
    anomaly_names=None,
    detectors=("START_beam_mod1","START_beam_mod2","RICH_trends","ToF_multiplicity","ToF_sum"),
    method="hdbscan",                 # see options above
    max_samples=2000,
    perplexity=30,
    random_state=42,
    standardize=False,                # keep False for encoder latents unless needed
    pca_components=None,              # int, or float in (0,1] for variance kept; None to skip PCA
    k_max=6,                          # used by spectral/bayes_gmm for #components max
    contamination=0.05,               # expected anomaly fraction for score-based methods
    # HDBSCAN/DBSCAN params
    min_cluster_size=25, min_samples=10, eps=0.8,
    # LOF/OCSVM params
    n_neighbors=20, nu=0.05,
    # KDE params
    bandwidth=None,
    # thresholding
    quantile_override=None,           # e.g. 0.95 (top 5% by score -> anomalies)
    use_evt=False, evt_tail_frac=0.05
    ):
    """
    Unsupervised per-detector anomaly detection with multiple methods.
    Left plot: TRUE labels (if available) for reference only.
    Right plot: predicted binary normal vs anomaly.
    Returns a concatenated DataFrame of results.
    """
    if anomaly_names is None:
        anomaly_names = {
            0:"no anomaly", 1:"empty bin", 2:"noisy bin", 3:"sparse bin",
            4:"comb_artifact", 5:"uniform noise", 6:"skewed distribution",
            7:"edge effects", 8:"secondary peak"
        }

    base = os.path.join(root_dir, f"logs/VAE/Model-{Model_version}",
                        f"{events}_hpt", f"{poa}_poa", f"version_{version}")
    out_dir = os.path.join(base, f"TSNE_{method.upper()}_PerDetector_Unsupervised")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    detector_metrics = []  # Store metrics for each detector

    for det in detectors:
        # ----- load features + (optional) labels for plotting only -----
        try:
            X = np.load(os.path.join(base, "latent_features", f"{det}_latent_features.npy"))
            y_path = os.path.join(base, "latent_features", f"{det}_labels.csv")
            y = pd.read_csv(y_path)["label"].values if os.path.exists(y_path) else None
        except FileNotFoundError:
            print(f"[{det}] missing latent_features, skipping.")
            continue

        n = len(X)
        if y is not None: n = min(n, len(y))
        if n > max_samples:
            idx = np.random.choice(n, max_samples, replace=False)
            X = X[idx]
            if y is not None: y = y[idx]
            n = len(X)
        elif y is not None:
            X, y = X[:n], y[:n]

        # ----- preprocess -----
        X_in = X.astype(np.float64)
        if standardize:
            X_in = StandardScaler().fit_transform(X_in)

        # Optional PCA (helps a lot for density/boundary methods)
        if pca_components is not None:
            if isinstance(pca_components, float) and 0 < pca_components <= 1.0:
                pca = PCA(n_components=pca_components, svd_solver="full", random_state=random_state)
            elif isinstance(pca_components, int) and pca_components >= 1:
                pca = PCA(n_components=pca_components, random_state=random_state)
            else:
                pca = None
            if pca is not None:
                X_in = pca.fit_transform(X_in)

        # ----- method-specific predictions (no labels used) -----
        pred_bin = None
        score = None
        cluster_labels = None
        info = ""
        accuracy = None  # Initialize accuracy

        if method == "hdbscan":
            if not _HAS_HDBSCAN:
                raise ImportError("hdbscan is not installed. pip install hdbscan")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=min_samples, metric="euclidean")
            clusterer.fit(X_in)
            cluster_labels = clusterer.labels_
            normal_c = _central_cluster(cluster_labels, X_in)
            # Start with: noise (-1) OR not-normal cluster = anomaly
            pred_bin = (cluster_labels == -1)
            if normal_c is not None:
                pred_bin = np.where(cluster_labels == normal_c, 0, 1)
            pred_bin = pred_bin.astype(int)
            # Optional score-based override using HDBSCAN outlier scores
            if hasattr(clusterer, "outlier_scores_"):
                score = clusterer.outlier_scores_
            else:
                # distance to chosen normal centroid as a fallback score
                if normal_c is not None:
                    mu_norm = X_in[cluster_labels == normal_c].mean(axis=0)
                    score = np.linalg.norm(X_in - mu_norm, axis=1)
            if quantile_override is not None or use_evt:
                if score is None:  # last resort
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
            # choose #clusters up to k_max; simple heuristic = min(5, k_max)
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
            # Higher decision_function -> more normal
            dfun = iso.decision_function(X_in)
            score = -dfun  # higher => more anomalous
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
            _ = lof.fit_predict(X_in)  # assigns labels internally
            # more negative => more anomalous; invert
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
            # decision_function > 0 => inlier
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
            # Scott's rule of thumb if bandwidth not provided
            if bandwidth is None:
                std = np.mean(X_in.std(axis=0))
                d = X_in.shape[1]
                bandwidth = std * (X_in.shape[0] ** (-1.0 / (d + 4)))
                bandwidth = max(bandwidth, 1e-3)
            kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(X_in)
            log_dens = kde.score_samples(X_in)
            score = -log_dens  # lower density => higher anomaly score
            if quantile_override is not None:
                thr = np.quantile(score, quantile_override)
            elif use_evt:
                thr = _evt_threshold(score, tail_frac=evt_tail_frac, q=0.99)
            else:
                thr = np.quantile(score, 1.0 - contamination)
            pred_bin = (score > thr).astype(int)
            info = f"bw={bandwidth:.3g}"

        elif method == "pca_mahal":
            # robust covariance in current space
            lw = LedoitWolf().fit(X_in)
            mu = lw.location_
            VI = lw.precision_
            diffs = X_in - mu
            # squared Mahalanobis distances
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

        # ----- Calculate accuracy if true labels are available -----
        if y is not None:
            # Convert true labels to binary (0 = normal, 1 = anomaly)
            # Assuming 'no anomaly' is label 0, all others are anomalies
            true_binary = np.where(y == 0, 0, 1)
            accuracy = accuracy_score(true_binary, pred_bin)
            info += f", Acc={accuracy:.3f}"
            
            # Store metrics for this detector
            detector_metrics.append({
                "detector": det,
                "accuracy": accuracy,
                "precision": precision_score(true_binary, pred_bin, zero_division=0),
                "recall": recall_score(true_binary, pred_bin, zero_division=0),
                "f1": f1_score(true_binary, pred_bin, zero_division=0)
            })

        # ----- t-SNE visualization -----
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        Z = tsne.fit_transform(X_in)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: TRUE labels (reference only)
        if y is not None:
            uniq = np.unique(y)
            cmap = plt.cm.tab20 if len(uniq) > 10 else plt.cm.tab10
            cols = cmap(np.linspace(0, 1, len(uniq)))
            for i, lbl in enumerate(uniq):
                m = (y == lbl)
                axes[0].scatter(Z[m,0], Z[m,1], c=[cols[i]], s=10, alpha=0.75,
                                label=anomaly_names.get(lbl, f"label {lbl}"))
            axes[0].legend(title="True Anomaly Type", bbox_to_anchor=(1.02, 1), loc="upper left")
            axes[0].set_title(f"{det} - True Anomaly Types")
        else:
            axes[0].scatter(Z[:,0], Z[:,1], s=8, alpha=0.7)
            axes[0].set_title(f"{det} - t-SNE (unlabeled)")

        axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2"); axes[0].grid(True, alpha=0.3)

        # Right: predicted binary
        for lab, col, val in [("Normal (Pred)", "green", 0), ("Anomaly (Pred)", "red", 1)]:
            m = (pred_bin == val)
            if np.any(m):
                axes[1].scatter(Z[m,0], Z[m,1], c=col, s=10, alpha=0.75, label=lab)
        
        # Include accuracy in the title if available
        title = f"{det} - {method.upper()} Prediction"
        if accuracy is not None:
            title = f"{det} - {method.upper()} Prediction (Acc={accuracy:.3f})"
        
        axes[1].set_title(title)
        axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2"); axes[1].grid(True, alpha=0.3)
        axes[1].legend(title="Predicted")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{det}_tsne_{method}.png"),
                    dpi=220, bbox_inches="tight")
        plt.close()

        # ----- collect results -----
    df = {
        "detector": [det] * n,
        "pred_binary": pred_bin,
        "tsne_x": Z[:, 0],
        "tsne_y": Z[:, 1],
        "method": [method] * n
    }

    if y is not None: 
        df["true_label"] = y
        # Calculate per-sample correctness (1 if correct, 0 if wrong) - NOT accuracy!
        true_binary = np.where(y == 0, 0, 1)
        df["correct"] = (pred_bin == true_binary).astype(int)
        
    if score is not None: 
        df["score"] = score
    if cluster_labels is not None: 
        df["cluster"] = cluster_labels

    all_results.append(pd.DataFrame(df))

    # Store metrics for this detector (overall performance)
    if y is not None:
        detector_metrics.append({
            "detector": det,
            "accuracy": accuracy,
            "precision": precision_score(true_binary, pred_bin, zero_division=0),
            "recall": recall_score(true_binary, pred_bin, zero_division=0),
            "f1": f1_score(true_binary, pred_bin, zero_division=0),
            "n_samples": n,
            "n_normal": np.sum(true_binary == 0),
            "n_anomaly": np.sum(true_binary == 1)
        })

    print(f"[{det}] {method} | saved plot. {info}")

    if not all_results:
        print("No detectors processed.")
        return None

    results = pd.concat(all_results, ignore_index=True)
    
    # Save detector metrics to a separate CSV
    if detector_metrics:
        metrics_df = pd.DataFrame(detector_metrics)
        metrics_df.to_csv(os.path.join(out_dir, f"detector_metrics_{method}.csv"), index=False)
        print(f"Metrics saved to: {os.path.join(out_dir, f'detector_metrics_{method}.csv')}")
    
    results.to_csv(os.path.join(out_dir, f"per_detector_{method}_unsupervised.csv"), index=False)
    print(f"Results saved to: {os.path.join(out_dir, f'per_detector_{method}_unsupervised.csv')}")
    return results

# ---------- shared helpers (label canonicalization + colors) ----------
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


# ==============================
# Figure 1
# ==============================
def per_detector_tsne_dataloader(
    root_dir, events,
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
    # HDBSCAN/DBSCAN
    min_cluster_size=25, min_samples=10, eps=0.8,
    # LOF/OCSVM
    n_neighbors=20, nu=0.05,
    # KDE
    bandwidth=None,
    # thresholding
    quantile_override=None,
    use_evt=False, evt_tail_frac=0.05
    ):
    """
    Unsupervised per-detector anomaly detection with multiple methods.
    Left plot: TRUE labels (if available) for reference only.
    Right plot: predicted binary normal vs anomaly.
    Returns a concatenated DataFrame of results.
    """

    # canonical anomaly names (spaces) + consistent color map
    if anomaly_names is None:
        anomaly_names = _default_anomaly_names()
    else:
        # sanitize incoming names (convert underscores to spaces)
        anomaly_names = {k: _canon_label(v, _default_anomaly_names()) for k, v in anomaly_names.items()}
    color_map = _fixed_color_map(anomaly_names)
    legend_order = [anomaly_names[k] for k in sorted(anomaly_names.keys())]  # "no anomaly" first

    base = os.path.join(root_dir, "HADES_t1_dataset")
    out_dir = os.path.join(base, f"TSNE_{method.upper()}_PerDetector_Unsupervised")
    labels_path = os.path.join(base, f"labels/labels_{events}_events.csv")
    numpy_path = os.path.join(base, f"numpy/{events}_events")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    detector_metrics = []

    for det in detectors:
        # ----- load features + (optional) labels for plotting only -----
        try:
            X = np.load(os.path.join(numpy_path, f"{det}.npy"))
            y = pd.read_csv(labels_path)[f"{det}_anomaly"].values if os.path.exists(labels_path) else None
        except FileNotFoundError:
            print(f"[{det}] missing data, skipping.")
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

        # ----- method-specific predictions (no labels used) -----
        pred_bin = None
        score = None
        cluster_labels = None
        info = ""
        accuracy = None

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

        # ----- metrics if labels available -----
        if y is not None and len(y) == n:
            y_canon = _canon_array(y, anomaly_names)
            true_binary = np.where(y_canon == "no anomaly", 0, 1)
            accuracy = accuracy_score(true_binary, pred_bin)
            detector_metrics.append({
                "detector": det,
                "accuracy": accuracy,
                "precision": precision_score(true_binary, pred_bin, zero_division=0),
                "recall": recall_score(true_binary, pred_bin, zero_division=0),
                "f1": f1_score(true_binary, pred_bin, zero_division=0)
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
        if accuracy is not None:
            title += f" (Acc={accuracy:.3f})"
        axes[1].set_title(title)
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(title="Predicted")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{det}_tsne_{method}.png"), dpi=220, bbox_inches="tight")
        plt.close()

        # ----- collect results -----
        df = {
            "detector": det,
            "pred_binary": pred_bin,
            "tsne_x": Z[:, 0],
            "tsne_y": Z[:, 1],
            "method": np.array([method] * n)
        }
        if y is not None and len(y) == n:
            df["true_label"] = y
            if accuracy is not None:
                df["accuracy"] = np.array([accuracy] * n)
        if score is not None:
            df["score"] = score
        if cluster_labels is not None:
            df["cluster"] = cluster_labels
        all_results.append(pd.DataFrame(df))

        print(f"[{det}] {method} | saved plot. {info}")

    if not all_results:
        print("No detectors processed.")
        return None

    results = pd.concat(all_results, ignore_index=True)

    if detector_metrics:
        metrics_df = pd.DataFrame(detector_metrics)
        metrics_df.to_csv(os.path.join(out_dir, f"detector_metrics_{method}.csv"), index=False)
        print(f"Metrics saved to: {os.path.join(out_dir, f'detector_metrics_{method}.csv')}")

    results.to_csv(os.path.join(out_dir, f"per_detector_{method}_unsupervised.csv"), index=False)
    print(f"Results saved to: {os.path.join(out_dir, f'per_detector_{method}_unsupervised.csv')}")
    return results
