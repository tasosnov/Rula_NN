# -*- coding: utf-8 -*-
"""
PC1–Epoch (Progress) clustering WITH SORTING
--------------------------------------------
Input:  ./PCA/*.csv
Output: ./results_kmeans_pc1_epoch/k{K}/...

Feature: Clusters are sorted by Time/Progress.
         Label 0 = Early life (Healthy)
         Label K-1 = End of life (Failure)
"""

import argparse
from pathlib import Path
import sys, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ----------------------------- I/O & detection -----------------------------

def detect_column(df: pd.DataFrame, candidates):
    for cand in candidates:
        pat = re.compile(re.escape(cand), re.IGNORECASE)
        for col in df.columns:
            if pat.search(col):
                return col
    return None

def load_all_devices(pca_dir: Path):
    files = sorted(pca_dir.glob("*.csv"))
    if not files:
        print(f"[ERROR] No CSV files found in {pca_dir}")
        sys.exit(1)

    datasets = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
            continue

        col_epoch = detect_column(df, ["epoch","row_index","index","time","t","step"])
        col_pc1   = detect_column(df, ["pc1","pc 1","pc1_score","pc1score","pc1_component","PC1"])
        
        if col_epoch is None or col_pc1 is None:
            print(f"[WARN] {f.name}: missing epoch or pc1. Skipping.")
            continue

        # Keep only needed columns
        sub = df[[col_epoch, col_pc1]].copy()
        sub.columns = ["epoch", "pc1"]

        # --- OUTLIER REMOVAL (IQR) ---
        # Optional: Remove extreme outliers to help MinMaxScaler
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        Q1 = sub['pc1'].quantile(0.25)
        Q3 = sub['pc1'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 2.5 * IQR
        upper = Q3 + 2.5 * IQR
        sub = sub[(sub['pc1'] >= lower) & (sub['pc1'] <= upper)]
        # -----------------------------

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sub["epoch"] = sub["epoch"].astype(int)
            except Exception:
                pass

        sub = sub.sort_values("epoch").reset_index(drop=True)
        sub.insert(0, "device_id", f.stem)
        datasets.append(sub)

    if not datasets:
        print("[ERROR] No valid datasets after parsing.")
        sys.exit(1)

    return datasets

# ----------------------------- progress & scaling -----------------------------

def per_device_progress(d: pd.DataFrame) -> pd.DataFrame:
    """Calculates progress (0->1) based on epoch."""
    dd = d.copy()
    e_min = float(dd["epoch"].min())
    e_max = float(dd["epoch"].max())
    denom = e_max - e_min
    if denom == 0.0:
        dd["progress_01"] = np.nan
    else:
        dd["progress_01"] = (dd["epoch"].astype(float) - e_min) / denom
        dd["progress_01"] = dd["progress_01"].clip(0.0, 1.0)
    return dd

def fit_minmax_per_device(datasets):
    scalers = {}
    for d in datasets:
        dev = d["device_id"].iloc[0]
        scalers[(dev, "pc1")] = MinMaxScaler().fit(d[["pc1"]].to_numpy())
    return scalers

def apply_scaling_per_device(datasets, scalers):
    out = []
    for d in datasets:
        dev = d["device_id"].iloc[0]
        dd = d.copy()
        sc_pc1 = scalers[(dev, "pc1")]
        dd["pc1_scaled"] = sc_pc1.transform(d[["pc1"]].to_numpy())[:,0]
        out.append(dd)
    return out

# ----------------------------- K-Means with SORTING -----------------------------

def kmeans_labels_sorted(X: np.ndarray, k: int, random_state: int):
    """
    Runs K-Means and then RELABELS clusters based on their Progress coordinate.
    X column 0: PC1
    X column 1: Progress
    """
    if X.shape[0] < k:
        return None, None, None
        
    # 1. Run standard K-Means
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    
    # 2. Sort centers based on Column 1 (Progress)
    # This gives us the indices of centers from lowest progress to highest
    sorted_idx = np.argsort(centers[:, 1]) 
    
    # 3. Create a mapping: Old_Label -> New_Sorted_Label
    # If sorted_idx is [2, 0, 1], it means:
    # Old Cluster 2 becomes New Cluster 0 (Start)
    # Old Cluster 0 becomes New Cluster 1 (Middle)
    # Old Cluster 1 becomes New Cluster 2 (End)
    map_dict = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(sorted_idx)}
    
    # 4. Apply mapping to labels
    new_labels = np.array([map_dict[l] for l in labels])
    
    # 5. Reorder centers to match the new labels (for plotting)
    sorted_centers = centers[sorted_idx]
    
    return km, new_labels, sorted_centers

# ----------------------------- plotting -----------------------------

def plot_clustering_space(df: pd.DataFrame, labels, centers, title: str, outpath: Path):
    """Plot PC1 (y) vs Progress (x) with sorted clusters."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    
    x = df["progress_01"].to_numpy()
    y = df["pc1_scaled"].to_numpy()
    
    if labels is None:
        ax.scatter(x, y, c="#8c8c8c", s=20, alpha=0.5, label="no k-means")
    else:
        # Use a colormap that shows progression (e.g., viridis or plasma)
        cmap = plt.get_cmap("viridis", int(labels.max()) + 1)
        
        # Scatter plot colored by label
        sc = ax.scatter(x, y, c=labels, cmap=cmap, s=20, alpha=0.8, edgecolor='none')
        
        # Add colorbar to show the order
        cbar = plt.colorbar(sc, ax=ax, ticks=range(int(labels.max()) + 1))
        cbar.set_label('Cluster ID (0=Start -> K=End)')

        # Plot Centers
        if centers is not None:
            # centers are [pc1, progress]. We plot (progress, pc1)
            ax.scatter(centers[:,1], centers[:,0], c='red', marker="X", s=150, edgecolors="black", label="Centers", zorder=10)
            # Add numbers to centers
            for i, c in enumerate(centers):
                ax.text(c[1], c[0], str(i), fontsize=12, fontweight='bold', color='white', ha='center', va='center')

    ax.set_xlabel("Progress (0 -> 1)")
    ax.set_ylabel("PC1 (Scaled)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="Number of clusters (Stages)")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    # Setup paths
    root = Path(__file__).resolve().parent            
    pca_dir = root / "PCA"                            
    out_dir = root / "results_kmeans_pc1_epoch" / f"k{args.k}"  
    
    # Create directories
    for sub in ["plots/per_device", "plots/combined", "tables/per_device", "tables/combined"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading datasets...")
    datasets_raw = load_all_devices(pca_dir)

    # 2. Add Progress
    datasets_prog = [per_device_progress(d) for d in datasets_raw]

    # 3. Scale PC1
    scalers = fit_minmax_per_device(datasets_prog)
    datasets = apply_scaling_per_device(datasets_prog, scalers)

    # 4. Combine for K-Means
    combo = pd.concat(datasets, ignore_index=True)
    valid_mask = combo["pc1_scaled"].notna() & combo["progress_01"].notna()
    
    # Data for clustering: [PC1, Progress]
    Xc = combo.loc[valid_mask, ["pc1_scaled", "progress_01"]].to_numpy()
    
    print(f"Running K-Means (k={args.k}) on {len(Xc)} points...")
    
    # --- RUN SORTED K-MEANS ---
    km_model, labels_sorted, centers_sorted = kmeans_labels_sorted(Xc, args.k, args.random_state)
    
    # Assign labels back
    combo["cluster_global"] = np.nan
    if km_model is not None:
        combo.loc[valid_mask, "cluster_global"] = labels_sorted.astype(int)

    # 5. Save Combined Results
    plot_clustering_space(combo, combo["cluster_global"], centers_sorted, 
                          f"Combined Data | K={args.k} (Sorted by Progress)", 
                          out_dir / "plots/combined/combined_clusters.png")
    
    combo.to_csv(out_dir / "tables/combined/combined.csv", index=False)

    # 6. Save Per-Device Results
    # Need to split combo back to devices to keep order
    print("Saving per-device results...")
    
    # Group by device_id from combo dataframe directly
    for dev_id, df_dev in combo.groupby("device_id"):
        # Sort by epoch just to be safe
        df_dev = df_dev.sort_values("epoch")
        
        # Save CSV
        df_dev.to_csv(out_dir / f"tables/per_device/{dev_id}.csv", index=False)
        
        # Plot
        # We use the global centers for reference
        labels_dev = df_dev["cluster_global"].to_numpy()
        plot_clustering_space(df_dev, labels_dev, centers_sorted,
                              f"{dev_id} | K={args.k} (Sorted)",
                              out_dir / f"plots/per_device/{dev_id}.png")

    print(f"\nDone! Results saved in: {out_dir}")

if __name__ == "__main__":
    main()