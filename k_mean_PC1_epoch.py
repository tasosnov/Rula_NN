# -*- coding: utf-8 -*-
"""
PC1–Epoch (Progress) clustering (Modified)

Input  (relative): ./PCA/*.csv
Output (per run):  ./results_kmeans_pc1_epoch/k{K}/...

Changes from original:
- Clustering features: [pc1_scaled, progress_01] instead of [pc1_scaled, pc2_scaled]
- Plots updated to show PC1 vs Progress as the main clustering space.
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
    """Return first column whose name contains any candidate (case-insensitive)."""
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
        # We still load PC2 if present, just in case, but won't use it for clustering
        col_pc2   = detect_column(df, ["pc2","pc 2","pc2_score","pc2score","pc2_component","PC2"])

        # Check minimally for epoch and pc1
        missing = [name for name, col in [("epoch", col_epoch), ("pc1", col_pc1)] if col is None]
        if missing:
            print(f"[WARN] {f.name}: missing required columns {missing}. Skipping.")
            continue

        cols_to_keep = [col_epoch, col_pc1]
        new_names = ["epoch", "pc1"]
        if col_pc2:
            cols_to_keep.append(col_pc2)
            new_names.append("pc2")
        
        sub = df[cols_to_keep].copy()
        sub.columns = new_names

        # Drop NaNs
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["epoch","pc1"])
        
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
    """progress_01 = (epoch - e_min)/(e_max - e_min)."""
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
    """Per-device MinMax for PC1 (and PC2 if exists)."""
    scalers = {}
    for d in datasets:
        dev = d["device_id"].iloc[0]
        # Scale PC1
        scalers[(dev, "pc1")] = MinMaxScaler().fit(d[["pc1"]].to_numpy())
        # Scale PC2 if exists
        if "pc2" in d.columns:
            scalers[(dev, "pc2")] = MinMaxScaler().fit(d[["pc2"]].to_numpy())
    return scalers

def apply_scaling_per_device(datasets, scalers):
    out = []
    for d in datasets:
        dev = d["device_id"].iloc[0]
        dd = d.copy()
        
        # Scale PC1
        sc_pc1 = scalers[(dev, "pc1")]
        dd["pc1_scaled"] = sc_pc1.transform(d[["pc1"]].to_numpy())[:,0]
        
        # Scale PC2 if exists
        if "pc2" in d.columns:
            sc_pc2 = scalers[(dev, "pc2")]
            dd["pc2_scaled"] = sc_pc2.transform(d[["pc2"]].to_numpy())[:,0]
        else:
            dd["pc2_scaled"] = np.nan
            
        out.append(dd)
    return out

# --------------------------------- k-means --------------------------------------

def kmeans_labels(X: np.ndarray, k: int, random_state: int):
    """Return (model, labels) or (None, None) if not enough points."""
    # X rows with NaN (e.g. single epoch progress) should be handled or dropped before
    # Here we assume X is clean.
    if X.shape[0] < k:
        return None, None
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels

# --------------------------------- plotting -------------------------------------

def plot_clustering_space(df: pd.DataFrame, labels, centers, title: str, outpath: Path):
    """
    Scatter plot of the clustering space: PC1 (scaled) vs Progress (0-1).
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=140)
    
    # X-axis: Progress (Epoch scaled), Y-axis: PC1 scaled
    x = df["progress_01"].to_numpy()
    y = df["pc1_scaled"].to_numpy()
    
    if labels is None:
        ax.scatter(x, y, c="#8c8c8c", s=28, alpha=0.9, edgecolors="none", label="no k-means")
    else:
        labels = np.asarray(labels)
        k = int(labels.max()) + 1 if labels.size else 0
        cmap = plt.get_cmap("tab10")
        for ci in range(k):
            idx = np.where(labels == ci)[0]
            if idx.size == 0:
                continue
            ax.scatter(x[idx], y[idx], s=28, alpha=0.9, edgecolors="none", c=[cmap(ci % 10)], label=f"cluster {ci}")
        
        # Plot centers if provided (centers are in [progress, pc1] or [pc1, progress] space?)
        # NOTE: We clustered on [pc1_scaled, progress_01], so centers are (pc1, progress).
        # But we plot x=progress, y=pc1. So we must swap coordinates for plotting centers.
        if centers is not None:
            # centers[:,0] is pc1, centers[:,1] is progress
            ax.scatter(centers[:,1], centers[:,0], marker="X", s=110, linewidths=1.2, edgecolors="black", label="centers")
            
    ax.legend(loc="best", frameon=True, title="k-means (global)")
    ax.set_xlabel("Progress (0→1)")
    ax.set_ylabel("PC1 (min–max scaled)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def plot_hi_inv_vs_rul(df: pd.DataFrame, title: str, outpath: Path):
    """Scatter y=1-pc1_scaled vs progress_01 and overlay RUL_line = 1 - progress_01."""
    if "progress_01" not in df.columns or df["progress_01"].isna().all():
        return
    x = df["progress_01"].to_numpy()
    hi_inv = 1.0 - df["pc1_scaled"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    ax.scatter(x, hi_inv, s=26, alpha=0.9, edgecolors="none", label="HI_inv = 1 - pc1_scaled")
    ax.plot([0,1], [1,0], linewidth=1.6, alpha=0.95, label="RUL_line = 1 - progress")
    ax.set_xlabel("per-device progress (0→1)"); ax.set_ylabel("value")
    ax.set_title(title); ax.grid(True, alpha=0.25); ax.legend(loc="best", frameon=True)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

# --------------------------------- fs layout ------------------------------------

def ensure_dirs(base: Path):
    subpaths = [
        "plots/per_device",
        "plots/per_device_hi_rul",
        "plots/combined",
        "tables/per_device",
        "tables/per_device_progress",
        "tables/per_device_hi_rul",
        "tables/combined",
        "tables/combined_progress",
        "tables/combined_hi_rul",
    ]
    for sub in subpaths:
        (base / sub).mkdir(parents=True, exist_ok=True)

# ----------------------------------- main ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PC1–Epoch(Progress) k-means.")
    ap.add_argument("--k", type=int, default=3, help="number of clusters")
    ap.add_argument("--random_state", type=int, default=42, help="random seed for k-means")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent            
    pca_dir = root / "PCA"                            
    # Changed output directory to distinguish from PC1-PC2 clustering
    out_dir = root / "results_kmeans_pc1_epoch" / f"k{args.k}"  
    ensure_dirs(out_dir)

    # 1) Load
    datasets_raw = load_all_devices(pca_dir)

    # 2) progress per device
    datasets_prog = [per_device_progress(d) for d in datasets_raw]

    # 3) per-device scaling
    scalers  = fit_minmax_per_device(datasets_prog)
    datasets = apply_scaling_per_device(datasets_prog, scalers)

    # 4) combined k-means (global labels) on [PC1, Progress]
    combo = pd.concat(datasets, ignore_index=True)
    
    # ---- KEY CHANGE HERE: Clustering on PC1 and Progress ----
    # Drop rows with NaN (e.g. if single epoch file resulted in NaN progress)
    valid_mask = combo["pc1_scaled"].notna() & combo["progress_01"].notna()
    combo_valid = combo.loc[valid_mask].copy()
    
    Xc = combo_valid[["pc1_scaled", "progress_01"]].to_numpy()
    
    kmc, labels_c = kmeans_labels(Xc, args.k, args.random_state)
    
    # Initialize global cluster column with NaN
    combo["cluster_global"] = np.nan
    centers_c = None
    
    if kmc is not None:
        combo.loc[valid_mask, "cluster_global"] = labels_c.astype(int)
        centers_c = kmc.cluster_centers_  # shape (k, 2) -> [pc1, progress]

    # 5) push labels back per device (keep concat order)
    per_device = []
    start = 0
    for d in datasets:
        n = len(d)
        dd = d.copy()
        dd["cluster_global"] = combo["cluster_global"].iloc[start:start+n].to_numpy()
        per_device.append(dd)
        start += n

    # 6) per-device outputs
    rows_hi_rul = []
    for d in per_device:
        dev = d["device_id"].iloc[0]

        # PLOT: PC1 vs Progress (Clustering Space)
        plot_clustering_space(
            d,
            d["cluster_global"].to_numpy() if not d["cluster_global"].isna().all() else None,
            centers_c,
            title=f"{dev} | PC1 vs Progress (Clusters) | k={args.k}",
            outpath=out_dir / f"plots/per_device/{dev}_clusters.png"
        )
        
        # Plot: HI_inv vs RUL_line
        plot_hi_inv_vs_rul(
            d, title=f"{dev} HI_inv vs RUL_line | k={args.k}",
            outpath=out_dir / f"plots/per_device_hi_rul/{dev}_hi_inv_vs_rul.png"
        )

        # Tables
        cols_export = ["device_id","epoch","progress_01","pc1_scaled","cluster_global"]
        if "pc2_scaled" in d.columns:
            cols_export.append("pc2_scaled")
            
        base = d[cols_export].copy()
        base["k"] = args.k
        base.to_csv(out_dir / f"tables/per_device/{dev}.csv", index=False)

        prog = d[["device_id","epoch","progress_01"]].copy()
        prog.to_csv(out_dir / f"tables/per_device_progress/{dev}_progress.csv", index=False)

        hi_tbl = d[["device_id","epoch","progress_01","pc1_scaled","cluster_global"]].copy()
        hi_tbl["HI_inv"]   = 1.0 - d["pc1_scaled"]
        hi_tbl["RUL_line"] = 1.0 - d["progress_01"]
        hi_tbl.to_csv(out_dir / f"tables/per_device_hi_rul/{dev}_hi_rul.csv", index=False)
        rows_hi_rul.append(hi_tbl)

    # 7) combined outputs
    # Plot combined PC1 vs Progress
    plot_clustering_space(
        combo,
        combo["cluster_global"].to_numpy() if not combo["cluster_global"].isna().all() else None,
        centers_c,
        title=f"COMBINED | PC1 vs Progress (Clusters) | k={args.k}",
        outpath=out_dir / f"plots/combined/combined_clusters.png"
    )

    tblc = combo.copy()
    tblc["k"] = args.k
    tblc.to_csv(out_dir / f"tables/combined/combined.csv", index=False)

    prog_c = combo[["device_id","epoch","progress_01"]].copy()
    prog_c.to_csv(out_dir / f"tables/combined_progress/combined_progress.csv", index=False)

    if rows_hi_rul:
        combo_hi = pd.concat(rows_hi_rul, ignore_index=True)
        combo_hi.to_csv(out_dir / f"tables/combined_hi_rul/combined_hi_rul.csv", index=False)

    print(f"\nDone. Outputs → {out_dir}  (parent: {out_dir.parent})")

if __name__ == "__main__":
    main()