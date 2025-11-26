# -*- coding: utf-8 -*-
"""
PC1–PC2 clustering (from scratch, single-pass)

Input  (relative): ./PCA/*.csv
Output (per run):  ./results_kmeans/k{K}/...

Steps
1) Load CSVs, detect columns (epoch, pc1, pc2), clean, sort.
2) Per-device progress_01 = (epoch - e_min)/(e_max - e_min).
3) Per-device MinMax scaling for PC1/PC2 (epoch scaled for completeness).
4) Combined k-means on ALL rows over [pc1_scaled, pc2_scaled] → cluster_global.
5) Propagate cluster_global back to each device row.
6) Save tables (per_device, per_device_progress, per_device_hi_rul, combined, combined_progress, combined_hi_rul).
7) Plots:
   • PC1–PC2 by cluster_global (+centers X)
   • PC1 vs progress_01 by cluster_global
   • HI_inv (=1-pc1_scaled) vs progress_01 with RUL_line (=1-progress_01)

Run examples (Windows cmd):
  py k_means_clustering.py --k 3 --random_state 42
  py k_means_clustering.py --k 2 --random_state 42
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
        col_pc2   = detect_column(df, ["pc2","pc 2","pc2_score","pc2score","pc2_component","PC2"])

        missing = [name for name, col in [("epoch", col_epoch), ("pc1", col_pc1), ("pc2", col_pc2)] if col is None]
        if missing:
            print(f"[WARN] {f.name}: missing required columns {missing}. Skipping.")
            continue

        sub = df[[col_epoch, col_pc1, col_pc2]].copy()
        sub.columns = ["epoch","pc1","pc2"]

        n0 = len(sub)
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["epoch","pc1","pc2"])
        if n0 - len(sub) > 0:
            print(f"[INFO] {f.name}: dropped {n0 - len(sub)} non-finite rows.")

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
        print(f"[INFO] {dd['device_id'].iloc[0]}: single epoch; progress_01 set to NaN.")
        return dd
    dd["progress_01"] = (dd["epoch"].astype(float) - e_min) / denom
    dd["progress_01"] = dd["progress_01"].clip(0.0, 1.0)
    return dd

def fit_minmax_per_device(datasets):
    """Per-device MinMax for (pc1, pc2) and for epoch (kept for completeness)."""
    scalers = {}
    for d in datasets:
        dev = d["device_id"].iloc[0]
        scalers[(dev, "pc")]    = MinMaxScaler().fit(d[["pc1","pc2"]].to_numpy())
        scalers[(dev, "epoch")] = MinMaxScaler().fit(d[["epoch"]].to_numpy())
    return scalers

def apply_scaling_per_device(datasets, scalers):
    out = []
    for d in datasets:
        dev = d["device_id"].iloc[0]
        sc_pc = scalers[(dev, "pc")]
        sc_ep = scalers[(dev, "epoch")]
        pcs = sc_pc.transform(d[["pc1","pc2"]].to_numpy())
        eps = sc_ep.transform(d[["epoch"]].to_numpy())
        dd = d.copy()
        dd["pc1_scaled"]   = pcs[:,0]
        dd["pc2_scaled"]   = pcs[:,1]
        dd["epoch_scaled"] = eps[:,0]
        out.append(dd)
    return out

# --------------------------------- k-means --------------------------------------

def kmeans_labels(X: np.ndarray, k: int, random_state: int):
    """Return (model, labels) or (None, None) if not enough points."""
    if X.shape[0] < k:
        return None, None
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels

# --------------------------------- plotting -------------------------------------

def plot_pc12_by_cluster(df: pd.DataFrame, labels, centers, title: str, outpath: Path):
    """
    PC1–PC2: χρώμα = cluster label (global), Χ = κέντρα από combined k-means.
    Αν labels=None → γκρι scatter χωρίς κέντρα.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=140)
    x = df["pc1_scaled"].to_numpy(); y = df["pc2_scaled"].to_numpy()
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
        if centers is not None:
            ax.scatter(centers[:,0], centers[:,1], marker="X", s=110, linewidths=1.2, edgecolors="black", label="centers")
        ax.legend(loc="best", frameon=True, title="k-means (global)")
    ax.set_xlabel("pc1 (min–max scaled)"); ax.set_ylabel("pc2 (min–max scaled)")
    ax.set_title(title); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def plot_pc1_vs_progress_clusters(df: pd.DataFrame, labels, title: str, outpath: Path):
    """PC1 vs progress_01 colored by cluster labels (global); grey if None."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    x = df["progress_01"]; y = df["pc1_scaled"]
    if labels is None or (isinstance(labels, np.ndarray) and np.isnan(labels).all()):
        ax.scatter(x, y, c="#8c8c8c", s=26, alpha=0.9, edgecolors="none", label="no k-means")
        ax.legend(loc="best", frameon=True)
    else:
        labels = np.asarray(labels)
        k = int(labels.max()) + 1 if labels.size else 0
        cmap = plt.get_cmap("tab10")
        for ci in range(k):
            idx = np.where(labels == ci)[0]
            if idx.size == 0: continue
            ax.scatter(x.iloc[idx], y.iloc[idx], s=26, alpha=0.9, edgecolors="none", c=[cmap(ci % 10)], label=f"cluster {ci}")
        ax.legend(loc="best", frameon=True, title="k-means clusters (global)")
    ax.set_xlabel("per-device progress (0→1)"); ax.set_ylabel("pc1 (min–max scaled)")
    ax.set_title(title); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

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
    ap = argparse.ArgumentParser(description="PC1–PC2 k-means + per-device progress + HI_inv/RUL (single-pass).")
    ap.add_argument("--k", type=int, default=3, help="number of clusters")
    ap.add_argument("--random_state", type=int, default=42, help="random seed for k-means")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent            # e.g., C:\Users\up107\Documents\Rula_NN
    pca_dir = root / "PCA"                            # ./PCA
    out_dir = root / "results_kmeans" / f"k{args.k}"  # ./results_kmeans/k{k}/
    ensure_dirs(out_dir)

    # 1) Load
    datasets_raw = load_all_devices(pca_dir)

    # 2) progress per device
    datasets_prog = [per_device_progress(d) for d in datasets_raw]

    # 3) per-device scaling
    scalers  = fit_minmax_per_device(datasets_prog)
    datasets = apply_scaling_per_device(datasets_prog, scalers)

    # 4) combined k-means (global labels)
    combo = pd.concat(datasets, ignore_index=True)
    Xc = combo[["pc1_scaled","pc2_scaled"]].to_numpy()
    kmc, labels_c = kmeans_labels(Xc, args.k, args.random_state)
    if kmc is None:
        print(f"[INFO] Combined: not enough points for k={args.k}; cluster_global will be NaN.")
        combo["cluster_global"] = np.nan
        centers_c = None
    else:
        combo["cluster_global"] = labels_c.astype(int)
        centers_c = kmc.cluster_centers_

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

        # plots: PC1–PC2 by cluster_global (+combined centers)
        plot_pc12_by_cluster(
            d,
            d["cluster_global"].to_numpy() if not d["cluster_global"].isna().all() else None,
            centers_c,
            title=f"{dev} | PC1–PC2 by cluster_global | k={args.k}",
            outpath=out_dir / f"plots/per_device/{dev}.png"
        )
        # PC1 vs progress_01 by cluster_global
        plot_pc1_vs_progress_clusters(
            d,
            d["cluster_global"].to_numpy() if not d["cluster_global"].isna().all() else None,
            title=f"{dev} pc1 vs progress (by cluster_global) | k={args.k}",
            outpath=out_dir / f"plots/per_device/{dev}_pc1_vs_progress_by_cluster.png"
        )
        # HI_inv vs RUL_line
        plot_hi_inv_vs_rul(
            d, title=f"{dev} HI_inv vs RUL_line | k={args.k}",
            outpath=out_dir / f"plots/per_device_hi_rul/{dev}_hi_inv_vs_rul.png"
        )

        # tables
        base = d[["device_id","epoch","progress_01","pc1_scaled","pc2_scaled","cluster_global"]].copy()
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
    plot_pc12_by_cluster(
        combo,
        combo["cluster_global"].to_numpy() if not combo["cluster_global"].isna().all() else None,
        centers_c,
        title=f"COMBINED | PC1–PC2 by cluster_global | k={args.k}",
        outpath=out_dir / f"plots/combined/combined.png"
    )
    plot_pc1_vs_progress_clusters(
        combo,
        combo["cluster_global"].to_numpy() if not combo["cluster_global"].isna().all() else None,
        title=f"COMBINED pc1 vs progress (by cluster_global) | k={args.k}",
        outpath=out_dir / f"plots/combined/combined_pc1_vs_progress_by_cluster.png"
    )

    tblc = combo[["device_id","epoch","progress_01","pc1_scaled","pc2_scaled","cluster_global"]].copy()
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
