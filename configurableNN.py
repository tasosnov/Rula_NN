# -*- coding: utf-8 -*-
"""
Pooled LSTM (PC1-only) with per-part min–max on BOTH PC1 and idx (idx=row index),
global pool & sort by scaled idx, and two plots in Fig.15 style:

Outputs in --out_dir:
- NN_ckpt.pt
- run_meta.json
- predictions_tail_pooled.csv
- training_history.png
- unified_fig15_scaled_cutpred.png  (Actual, Training=fitted in-sample 1-step, Predictive=free-run after cut)
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------- optional GUI -------------
def pick_files_gui(multiple: bool = True, title: str = "Select CSV file(s)") -> List[str]:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
    if multiple:
        paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    else:
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        paths = [path] if path else []
    root.update()
    root.destroy()
    return list(paths)

# -------------------------
# Utils
# -------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def minmax_scale_whole(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    rng = xmax - xmin if xmax > xmin else 1.0
    x_scaled = (x - xmin) / rng
    return x_scaled.astype(np.float32), {"xmin": xmin, "xmax": xmax}


# -------------------------
# Dataset over pooled S
# -------------------------

class SlidingWindowSeries(Dataset):
    def __init__(self, series_1d: np.ndarray, lo: int, lp: int):
        assert series_1d.ndim == 1, "series must be 1-D"
        self.s = series_1d.astype(np.float32)
        self.lo = int(lo)
        self.lp = int(lp)
        self.N = len(self.s)
        self.length = max(0, self.N - (self.lo + self.lp) + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        s = idx
        e = idx + self.lo
        y = idx + self.lo
        z = idx + self.lo + self.lp
        x_win = self.s[s:e]                 # shape (lo,)
        y_win = self.s[y:z]                 # shape (lp,)
        return torch.from_numpy(x_win).unsqueeze(-1), torch.from_numpy(y_win)


# -------------------------
# Model
# -------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, lp: int, dropout: float = 0.0):
        super().__init__()
        self.lp = lp
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Linear(hidden_size, lp)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)          # out: [B, Lo, H], h_n: [num_layers, B, H]
        last_hidden = h_n[-1]                    # [B, H]
        y = self.head(last_hidden)               # [B, Lp]
        return y


# -------------------------
# Train / Predict
# -------------------------

@dataclass
class TrainConfig:
    lo: int
    lp: int
    num_layers: int
    hidden_size: int
    input_size: int = 1


def save_checkpoint(path: str, model: nn.Module, config: TrainConfig,
                    scalers_pc1: Dict[str, Dict[str, float]], scalers_idx: Dict[str, Dict[str, float]],
                    extra: Dict = None):
    obj = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "scalers_pc1_per_file": scalers_pc1,  # {basename: {xmin,xmax}}
        "scalers_idx_per_file": scalers_idx,  # {basename: {xmin,xmax}}
        "extra": extra or {},
    }
    torch.save(obj, path)


def load_checkpoint(path: str, device: torch.device):
    obj = torch.load(path, map_location=device)
    return obj["state_dict"], obj["config"], obj.get("scalers_pc1_per_file", {}), obj.get("scalers_idx_per_file", {})


def check_compat(cfg_saved: dict, cfg_current: TrainConfig):
    errs = []
    for k in ("input_size","hidden_size","num_layers","lp","lo"):
        if cfg_saved.get(k) != getattr(cfg_current, k):
            errs.append(f"{k} ckpt={cfg_saved.get(k)} current={getattr(cfg_current,k)}")
    if errs:
        raise RuntimeError("Checkpoint/model config mismatch:\n  - " + "\n  - ".join(errs))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    loss_fn = nn.MSELoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict_multi(model, context: np.ndarray, lo: int, lp: int, device: torch.device) -> np.ndarray:
    """Return single Lp-step prediction given last Lo points (context)."""
    assert len(context) >= lo, "context shorter than Lo"
    x = context[-lo:].astype(np.float32).reshape(1, lo, 1)
    xb = torch.from_numpy(x).to(device)
    pred = model(xb).cpu().numpy().reshape(-1)
    return pred


def free_run_predict(model, S: np.ndarray, cut_idx: int, lo: int, lp: int, device: torch.device) -> np.ndarray:
    """Recursive multi-step prediction after cut_idx (feed predictions back)."""
    ctx = S[:cut_idx].copy()
    preds = []
    need = len(S) - cut_idx
    while len(preds) < need:
        yhat = predict_multi(model, ctx, lo, lp, device)
        preds.extend(yhat.tolist())
        ctx = np.concatenate([ctx, yhat.astype(np.float32)])
    return np.array(preds[:need], dtype=np.float32)


@torch.no_grad()
def in_sample_one_step_series(model, S: np.ndarray, upto_idx: int, lo: int, lp: int, device: torch.device) -> tuple:
    """
    In-sample **1-step-ahead** predictions on S[:upto_idx].
    Returns (x_idx, yhat), where x_idx = range(Lo, upto_idx), yhat[i] predicts S[i] from S[i-Lo:i].
    Uses only the first element of the Lp-vector.
    """
    xs = []
    yh = []
    for t in range(lo, max(lo, upto_idx)):
        ctx = S[t-lo:t]
        yhat = predict_multi(model, ctx, lo, lp, device)[0]  # 1-step ahead
        xs.append(t)
        yh.append(float(yhat))
    return np.array(xs, dtype=int), np.array(yh, dtype=np.float32)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="", help="CSV path or comma-separated list. Ignored if --gui_files.")
    ap.add_argument("--gui_files", action="store_true", help="Open GUI file picker (multi-select).")
    ap.add_argument("--column", type=str, default="PC1", help="Numeric column for input (PC1-only).")
    ap.add_argument("--lo", type=int, default=16, help="Observation window length.")
    ap.add_argument("--lp", type=int, default=1, help="Prediction horizon length.")
    ap.add_argument("--num_layers", type=int, default=1, help="LSTM layers.")
    ap.add_argument("--hidden_size", type=int, default=64, help="Hidden units per layer.")
    ap.add_argument("--dropout", type=float, default=0.0, help="LSTM dropout (only if num_layers>1).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--checkpoint", type=str, default="NN_ckpt.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="NN_out", help="Output directory for artifacts.")
    ap.add_argument("--cut_ratio", type=float, default=0.8, help="Cut position as fraction of pooled S length (Fig.15 style).")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Resolve files
    if args.gui_files:
        files = pick_files_gui(multiple=True, title="Select CSV file(s)")
        if not files:
            raise SystemExit("No files selected.")
    else:
        if not args.data_csv:
            raise SystemExit("Provide --data_csv or use --gui_files.")
        files = [p.strip() for p in args.data_csv.split(",") if p.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, args.checkpoint)
    meta_path = os.path.join(args.out_dir, "run_meta.json")

    # Read & per-part min–max scale for PC1 and idx (idx := row index 0..N-1)
    pooled_rows = []
    scalers_pc1: Dict[str, Dict[str, float]] = {}
    scalers_idx: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for path in files:
        base = os.path.basename(path)
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

        # Resolve PC1 column
        col = args.column if args.column in df.columns else None
        if col is None:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError(f"[{base}] No numeric columns and '{args.column}' not found.")
            col = num_cols[0]

        pc1 = df[col].astype(float).to_numpy()

        # Row index as idx (float)
        idx = np.arange(len(df), dtype=float)

        # Per-part min–max for PC1 and idx (whole file)
        pc1_s, sc_pc1 = minmax_scale_whole(pc1)
        idx_s,  sc_idx = minmax_scale_whole(idx)
        scalers_pc1[base] = sc_pc1
        scalers_idx[base] = sc_idx

        # Original order for stable tie-break
        order = np.arange(len(pc1_s), dtype=np.int64)

        part_df = pd.DataFrame({
            "part": base,
            "idx_s": idx_s,
            "pc1_s": pc1_s,
            "ord": order
        })
        pooled_rows.append(part_df)
        counts[base] = len(part_df)

    pooled = pd.concat(pooled_rows, ignore_index=True)

    # Global sort: by scaled idx asc, then part name, then original order
    pooled.sort_values(by=["idx_s", "part", "ord"], ascending=[True, True, True],
                       inplace=True, kind="mergesort")
    pooled.reset_index(drop=True, inplace=True)

    # Build the single global sequence S from pooled pc1_s
    S = pooled["pc1_s"].to_numpy().astype(np.float32)
    N = len(S)
    if N < args.lo + args.lp + 1:
        raise ValueError("Series too short for the selected Lo/Lp.")

    # Dataset over S (use ALL samples, no split)
    ds = SlidingWindowSeries(S, lo=args.lo, lp=args.lp)
    if len(ds) == 0:
        raise ValueError("No usable windows after pooling; adjust --lo/--lp or data length.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    cfg = TrainConfig(lo=args.lo, lp=args.lp, num_layers=args.num_layers, hidden_size=args.hidden_size, input_size=1)
    model = LSTMForecaster(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, lp=args.lp, dropout=0.0).to(device)

    # Train & record losses (use ALL windows)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_losses: List[float] = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, loader, optimizer, device)
        train_losses.append(tr_loss)
        print(f"Epoch {epoch:03d} | train MSE={tr_loss:.6f}")

    # Save artifacts
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print("Saved checkpoint to:", ckpt_path)

    meta = {
        "files": [os.path.basename(p) for p in files],
        "config": asdict(cfg),
        "scalers_pc1_per_file": scalers_pc1,
        "scalers_idx_per_file": scalers_idx,
        "pooled_length": int(N),
        "order_by": ["idx_s", "part", "ord"],
        "idx_definition": "row index 0..N-1 per file",
        "cut_ratio": args.cut_ratio
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Wrote metadata to:", meta_path)

    # --------- Plot 1: training_history.png ---------
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(1, len(train_losses)+1), train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training History (Unified PC1-only)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_history.png"), dpi=150)
    plt.close()

    # --------- Plot 2: unified_fig15_scaled_cutpred.png ---------
    cut_idx = int(N * args.cut_ratio)
    cut_idx = max(args.lo, min(N-args.lp, cut_idx))

    # In-sample 1-step-ahead over training segment
    x_fit, y_fit = in_sample_one_step_series(model, S, cut_idx, args.lo, args.lp, device)

    # Free-run predictions after cut (scaled)
    preds_scaled = free_run_predict(model, S, cut_idx, args.lo, args.lp, device)

    # Save tail preds as CSV
    np.savetxt(os.path.join(args.out_dir, "predictions_tail_pooled.csv"),
               preds_scaled.reshape(1, -1), delimiter=",",
               header=",".join([f"t+{i+1}" for i in range(len(preds_scaled))]), comments="")

    # Build figure
    x = np.arange(N)
    plt.figure(figsize=(12,5))
    plt.scatter(x, S, s=8, alpha=0.6, label="Actual value")
    if len(x_fit) > 0:
        plt.plot(x_fit, y_fit, linewidth=2.0, label="Training value")
    x_pred = np.arange(cut_idx, cut_idx + len(preds_scaled))
    plt.scatter(x_pred, preds_scaled, s=12, marker='x', label="Predictive value")
    plt.axvline(cut_idx, linestyle='--', linewidth=1.5)
    plt.xlabel("Serial number of the characteristic point (pooled order)")
    plt.ylabel("PC1 (scaled 0–1 per part)")
    plt.title("Actual, Training (fitted 1-step), and Predictive (free-run)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "unified_fig15_scaled_cutpred.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
