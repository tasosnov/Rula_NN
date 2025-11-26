# -*- coding: utf-8 -*-
"""
NN_softpi_alpha.py
- ΜΟΝΟ Soft-PI: pi_alpha για μονοτονία μέσα στο Lp block (0 => OFF, >0 => ON)
- Per-part min–max για PC1 και idx (row index)
- Pooling όλων των parts και ταξινόμηση με βάση scaled idx
- LSTM forecasting με Lo, Lp, hidden size, layers ρυθμιζόμενα
- Fig15-style plot: Actual (scatter), Training 1-step (line), Predictive free-run (x)
- Εμφάνιση βασικών ρυθμίσεων στα plots (χωρίς out_dir)

Απαιτήσεις: numpy, pandas, matplotlib, torch
"""

from __future__ import annotations
import argparse, os, json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- Βοηθητικά --------------------
def pick_files_gui(multiple: bool = True, title: str = "Select CSV file(s)") -> List[str]:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.update()
    if multiple:
        paths = filedialog.askopenfilenames(title=title, filetypes=[("CSV files","*.csv"),("All files","*.*")])
    else:
        path = filedialog.askopenfilename(title=title, filetypes=[("CSV files","*.csv"),("All files","*.*")])
        paths = [path] if path else []
    try:
        root.destroy()
    except Exception:
        pass
    return list(paths)


# === Soft-PI penalty: boundary (mean vs previous mean) + intra-block (drops) ===
import torch as _torch
def pi_penalty(pred: _torch.Tensor, prev_ref: _torch.Tensor | None, reduction: str = "mean") -> _torch.Tensor:
    B, Lp = pred.shape
    terms = []
    if prev_ref is not None:
        curr_mean = pred.mean(dim=1)
        terms.append(_torch.relu(prev_ref - curr_mean))  # punish mean drop
    if Lp >= 2:
        drops = pred[:, :-1] - pred[:, 1:]
        terms.append(_torch.relu(drops).view(B, -1))
    if not terms:
        return pred.new_zeros(())
    P = _torch.cat([t.view(B, -1) for t in terms], dim=1)
    if reduction == "mean":
        return P.mean()
    elif reduction == "sum":
        return P.sum()
    return P

    root = tk.Tk(); root.withdraw()
    ftypes=[("CSV files","*.csv"),("All files","*.*")]
    paths = filedialog.askopenfilenames(title=title, filetypes=ftypes) if multiple else \
            [filedialog.askopenfilename(title=title, filetypes=ftypes)]
    root.update(); root.destroy()
    return list(paths) if paths else []

def set_seed(s:int):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def minmax_scale_whole(x: np.ndarray) -> Tuple[np.ndarray, Dict[str,float]]:
    xmin=float(np.min(x)); xmax=float(np.max(x)); rng=xmax-xmin if xmax>xmin else 1.0
    return ((x-xmin)/rng).astype(np.float32), {"xmin":xmin,"xmax":xmax}

@dataclass
class TrainConfig:
    lo:int; lp:int; num_layers:int; hidden_size:int; input_size:int=1

class SlidingWindowSeries(Dataset):
    def __init__(self, s: np.ndarray, lo:int, lp:int):
        self.s=s.astype(np.float32); self.lo=int(lo); self.lp=int(lp)
        self.N=len(self.s); self.length=max(0, self.N-(self.lo+self.lp)+1)
    def __len__(self): return self.length
    def __getitem__(self, i:int):
        x=self.s[i:i+self.lo]
        y=self.s[i+self.lo:i+self.lo+self.lp]
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size:int, hidden:int, layers:int, lp:int, dropout:float=0.0):
        super().__init__(); self.lp=lp
        self.lstm=nn.LSTM(input_size, hidden, layers, batch_first=True,
                          dropout=(dropout if layers>1 else 0.0))
        self.head=nn.Linear(hidden, lp)
    def forward(self, x):
        out,(h,_)=self.lstm(x)   # h: [L,B,H]
        h_last=h[-1]             # [B,H]
        return self.head(h_last) # [B,Lp]

def pi_alpha_penalty(yhat: torch.Tensor) -> torch.Tensor:
    """
    Ποινή μονοτονίας εντός block: θέλουμε yhat[:, i+1] - yhat[:, i] >= 0
    """
    if yhat.size(1) <= 1:
        return yhat.new_tensor(0.0)
    diffs = yhat[:,1:] - yhat[:,:-1]        # θέλουμε diffs >= 0
    return F.relu(-diffs).mean()             # ποινή μόνο στις αρνητικές διαφορές

@torch.no_grad()
def predict_multi(model, context: np.ndarray, lo:int, lp:int, device) -> np.ndarray:
    x=context[-lo:].astype(np.float32).reshape(1,lo,1)
    return model(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)

def free_run_predict(model, S: np.ndarray, cut_idx:int, lo:int, lp:int, device) -> np.ndarray:
    ctx=S[:cut_idx].copy(); preds=[]; need=len(S)-cut_idx
    while len(preds)<need:
        yhat=predict_multi(model, ctx, lo, lp, device)
        preds.extend(yhat.tolist()); ctx=np.concatenate([ctx, yhat.astype(np.float32)])
    return np.array(preds[:need], dtype=np.float32)

def free_run_predict_extended(model, S: np.ndarray, cut_idx:int, lo:int, lp:int,
                              device, max_factor:int = 3) -> np.ndarray:
    """Free-run προβλέψεις μέχρι max_factor * N epochs συνολικά.
    Ξεκινάμε από cut_idx και συνεχίζουμε autoregressive,
    ακόμη και πέρα από το τελευταίο παρατηρημένο sample.
    Επιστρέφει προβλέψεις για t = cut_idx .. T_last,
    όπου T_last <= max_factor * N - 1.
    """
    N = len(S)
    if N <= 0:
        return np.zeros(0, dtype=np.float32)
    max_len = max_factor * N - cut_idx
    if max_len <= 0:
        return np.zeros(0, dtype=np.float32)

    ctx = S[:cut_idx].copy()
    preds: list[float] = []
    while len(preds) < max_len:
        yhat = predict_multi(model, ctx, lo, lp, device)
        preds.extend(yhat.tolist())
        ctx = np.concatenate([ctx, yhat.astype(np.float32)])
    return np.array(preds[:max_len], dtype=np.float32)

@torch.no_grad()
def in_sample_one_step_series(model, S: np.ndarray, upto_idx:int, lo:int, lp:int, device):
    xs=[]; yh=[]
    for t in range(lo, max(lo,upto_idx)):
        ctx=S[t-lo:t]; yhat=predict_multi(model, ctx, lo, lp, device)[0]
        xs.append(t); yh.append(float(yhat))
    return np.array(xs,dtype=int), np.array(yh,dtype=np.float32)


# -------------------- Main --------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="")
    ap.add_argument("--gui_files", action="store_true")
    ap.add_argument("--column", type=str, default="PC1")
    ap.add_argument("--lo", type=int, default=16)
    ap.add_argument("--lp", type=int, default=1)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="NN_out")
    ap.add_argument("--cut_ratio", type=float, default=0.8)
    ap.add_argument("--eol_thr", type=float, default=0.98, help="Threshold on HI/PC1 to define EoL (default 0.98)")
    # ΜΟΝΟ pi_alpha
    ap.add_argument("--pi_alpha", type=float, default=0.0, help="Weight for monotonicity penalty inside Lp block.")
    args=ap.parse_args()

    set_seed(args.seed); device=torch.device(args.device)
    if args.gui_files:
        files=pick_files_gui(True,"Select CSV file(s)")
        if not files: raise SystemExit("No files selected.")
    else:
        if not args.data_csv: raise SystemExit("Provide --data_csv or use --gui_files.")
        files=[p.strip() for p in args.data_csv.split(",") if p.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Φόρτωση & per-part min–max & pooling (sorted by scaled idx) ----
    pooled_rows=[]; scalers_pc1={}; scalers_idx={}
    for path in files:
        base=os.path.basename(path)
        df=pd.read_csv(path)
        df.columns=[str(c).strip() for c in df.columns]
        col=args.column if args.column in df.columns else \
             [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][0]
        pc1=df[col].astype(float).to_numpy()
        idx=np.arange(len(df), dtype=float)
        pc1_s, sc1=minmax_scale_whole(pc1)
        idx_s, sci=minmax_scale_whole(idx)
        scalers_pc1[base]=sc1; scalers_idx[base]=sci
        pooled_rows.append(pd.DataFrame({
            "part":base, "idx_s":idx_s, "pc1_s":pc1_s, "ord":np.arange(len(pc1_s))
        }))

    pooled=pd.concat(pooled_rows, ignore_index=True)
    pooled.sort_values(by=["idx_s","part","ord"], inplace=True, kind="mergesort")
    pooled.reset_index(drop=True, inplace=True)

    S=pooled["pc1_s"].to_numpy().astype(np.float32)
    N=len(S)

    ds=SlidingWindowSeries(S, args.lo, args.lp)
    if len(ds)==0: raise ValueError("No usable windows; adjust Lo/Lp.")
    loader=DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model=LSTMForecaster(1, args.hidden_size, args.num_layers, args.lp, args.dropout).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn=nn.MSELoss()

    # ---- Config string για annotation στα plots (χωρίς out_dir) ----
    pi_mode = "No PI" if args.pi_alpha == 0 else f"Soft-PI (alpha={args.pi_alpha:g})"
    CFG_STR = (f"Lo={args.lo}, Lp={args.lp}, layers={args.num_layers}, H={args.hidden_size}, "
               f"epochs={args.epochs}, cut={args.cut_ratio:.2f}, {pi_mode}")

    # ---- Training ----
    losses=[]
    prev_ref = None
    for ep in range(1, args.epochs+1):
        model.train(); tot=0.0; n=0
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            pred=model(xb)                     # [B,Lp]
            mse=loss_fn(pred,yb)
            mono = pi_penalty(pred, prev_ref=prev_ref, reduction="mean")        # μόνο εντός-block μονοτονία
            loss = mse + args.pi_alpha * mono
            opt.zero_grad(); loss.backward(); opt.step()
        # update boundary reference with current block mean
        with torch.no_grad():
            prev_ref = pred.mean().detach()
            tot += float(loss.item()) * xb.size(0); n += xb.size(0)
        losses.append(tot/max(n,1))
        print(f"Epoch {ep:03d} | Loss(total)={losses[-1]:.6f}")

    # ---- Evaluation on full dataset (sliding windows) & JSON export ----
    model.eval()
    abs_errors = []
    sq_errors = []
    with torch.no_grad():
        eval_loader = DataLoader(ds, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)
        for xb, yb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)            # [B, Lp]
            err = pred - yb             # [B, Lp]
            abs_errors.append(torch.abs(err).view(-1))
            sq_errors.append((err ** 2).view(-1))

    if abs_errors:
        abs_all = torch.cat(abs_errors)
        sq_all = torch.cat(sq_errors)
        test_mae = float(abs_all.mean().cpu().item())
        test_mse = float(sq_all.mean().cpu().item())
        test_rmse = float(torch.sqrt(sq_all.mean()).cpu().item())
    else:
        # fallback σε περίπτωση πολύ μικρής σειράς
        test_mae = float("nan")
        test_mse = float("nan")
        test_rmse = float("nan")

    # ---- Tail error AFTER cut (free-run vs actual) ----
    # Χρησιμοποιούμε το ίδιο cut_ratio και την free_run_predict.
    # Μετράμε το σφάλμα μόνο για t από cut_idx μέχρι N-1, όπου έχουμε ground truth.
    try:
        N = len(S)
        cut_idx_tail = int(N * args.cut_ratio)
        cut_idx_tail = max(args.lo, min(N - args.lp, cut_idx_tail))
        if cut_idx_tail < N:
            preds_tail = free_run_predict(model, S, cut_idx_tail,
                                          args.lo, args.lp, device)
            # βεβαιωνόμαστε ότι δεν ξεφεύγουμε από το πραγματικό tail
            tail_len = min(len(preds_tail), N - cut_idx_tail)
            if tail_len > 0:
                true_tail = S[cut_idx_tail:cut_idx_tail + tail_len]
                err_tail = preds_tail[:tail_len] - true_tail
                abs_tail = np.abs(err_tail)
                sq_tail = err_tail ** 2
                tail_mae = float(abs_tail.mean())
                tail_mse = float(sq_tail.mean())
                tail_rmse = float(np.sqrt(sq_tail.mean()))
            else:
                tail_mae = float("nan")
                tail_mse = float("nan")
                tail_rmse = float("nan")
        else:
            tail_mae = float("nan")
            tail_mse = float("nan")
            tail_rmse = float("nan")
    except Exception as _e:
        # σε περίπτωση απρόβλεπτου σφάλματος, δεν μπλοκάρουμε το script
        print("[WARN] tail metrics computation failed:", _e)
        tail_mae = float("nan")
        tail_mse = float("nan")
        tail_rmse = float("nan")

    # --- Predicted EoL epoch based on configurable threshold (args.eol_thr) ---
    thr_eol = float(getattr(args, "eol_thr", 0.98))

    pred_eol_epoch = None
    pred_eol_reached = False
    try:
        N_pred = len(S)
        cut_idx_tail = int(N_pred * args.cut_ratio)
        cut_idx_tail = max(args.lo, min(N_pred - args.lp, cut_idx_tail))
        if cut_idx_tail < N_pred:
            # free-run μέχρι και 3 * N epochs για EoL detection
            preds_tail = free_run_predict_extended(model, S, cut_idx_tail,
                                                   args.lo, args.lp, device,
                                                   max_factor=3)
            for k, val in enumerate(preds_tail):
                if val >= thr_eol:
                    pred_eol_epoch = int(cut_idx_tail + k)
                    pred_eol_reached = True
                    break
    except Exception as _e:
        print("[WARN] predicted EoL computation failed:", _e)
        pred_eol_epoch = None
        pred_eol_reached = False

    metrics = {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_mse": test_mse,
        "tail_mae": tail_mae,
        "tail_rmse": tail_rmse,
        "tail_mse": tail_mse,
        "pred_eol_epoch": pred_eol_epoch,
        "pred_eol_reached": pred_eol_reached,
        "eol_threshold": thr_eol,
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[METRICS]")
    print(json.dumps(metrics, indent=2))

    # ---- Plot 1: training_history.png ----
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(1,len(losses)+1), losses, marker='o')
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE + pi_alpha*mono)")
    plt.title("Training History — " + ("No PI" if args.pi_alpha == 0 else "Soft-PI (pi_alpha only)"))
    # settings box
    plt.gcf().text(0.995, 0.01, CFG_STR, ha="right", va="bottom", fontsize=9, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="0.6"))
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"training_history.png"), dpi=150)
    plt.close()

    # ---- Plot 2: unified_fig15_scaled_cutpred.png ----
    cut_idx=int(N*args.cut_ratio); cut_idx=max(args.lo, min(N-args.lp, cut_idx))
    x_fit, y_fit = in_sample_one_step_series(model, S, cut_idx, args.lo, args.lp, device)

    # Σύντομο free-run μέχρι το τέλος για CSV (όπως πριν)
    preds = free_run_predict(model, S, cut_idx, args.lo, args.lp, device)
    np.savetxt(os.path.join(args.out_dir,"predictions_tail_pooled.csv"),
               preds.reshape(1,-1), delimiter=",",
               header=",".join([f"t+{i+1}" for i in range(len(preds))]), comments="")

    # Extended free-run μέχρι 3N για plotting & predicted EoL
    preds_ext = free_run_predict_extended(model, S, cut_idx, args.lo, args.lp, device, max_factor=3)

    # Βρίσκουμε predicted EoL πάνω στην extended ουρά
    thr_eol_plot = float(getattr(args, "eol_thr", 0.98))
    pred_eol_epoch_plot = None
    for k, val in enumerate(preds_ext):
        if val >= thr_eol_plot:
            pred_eol_epoch_plot = int(cut_idx + k)
            break

    # Αποφασίζουμε μέχρι ποιο epoch θα δείξουμε το plot:
    # - Αν δεν υπάρχει predicted fail ή αν είναι <= N-1 → μέχρι N-1
    # - Αν predicted fail > N-1 → μέχρι pred_eol_epoch_plot
    if (pred_eol_epoch_plot is None) or (pred_eol_epoch_plot <= N-1):
        max_epoch_plot = N - 1
    else:
        max_epoch_plot = pred_eol_epoch_plot

    # Ετοιμάζουμε figure
    x = np.arange(N)
    plt.figure(figsize=(12,5))
    # Πραγματικές τιμές μόνο μέχρι N-1
    plt.scatter(x, S, s=8, alpha=0.6, label="Actual value")
    if len(x_fit)>0:
        plt.plot(x_fit, y_fit, linewidth=2.0, label="Training value")

    # Πόσα predicted points θα δείξουμε;
    if pred_eol_epoch_plot is None or pred_eol_epoch_plot <= N-1:
        tail_len_plot = min(len(preds_ext), N - cut_idx)
    else:
        tail_len_plot = min(len(preds_ext), max_epoch_plot - cut_idx + 1)

    if tail_len_plot > 0:
        x_pred = np.arange(cut_idx, cut_idx + tail_len_plot)
        plt.scatter(x_pred, preds_ext[:tail_len_plot], s=12, marker='x', label="Predictive value")

    # Cut line (έναρξη free-run)
    plt.axvline(cut_idx, linestyle='--', linewidth=1.5, label="Cut index")

    # Predicted EoL γραμμές, αν υπάρχει
    if pred_eol_epoch_plot is not None:
        plt.axvline(pred_eol_epoch_plot, linestyle='-.', linewidth=1.5,
                    label=f"Predicted EoL (thr={thr_eol_plot:.2f})")
        plt.axhline(thr_eol_plot, linestyle=':', linewidth=1.0)

    plt.xlim(0, max_epoch_plot)

    plt.xlabel("Serial number of the characteristic point (pooled order)")
    plt.ylabel("PC1 (scaled 0–1 per part)")
    plt.title("Actual, Training (fitted 1-step), and Predictive (free-run) — "
              + ("No PI" if args.pi_alpha == 0 else "Soft-PI (pi_alpha only)"))
    # settings box
    plt.gcf().text(0.995, 0.01, CFG_STR, ha="right", va="bottom", fontsize=9, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="0.6"))
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"unified_fig15_scaled_cutpred.png"), dpi=150)
    plt.close()

    # ---- Meta ----
    meta={"files":[os.path.basename(p) for p in files],
          "config":{"lo":args.lo,"lp":args.lp,"num_layers":args.num_layers,"hidden_size":args.hidden_size,
                    "epochs":args.epochs,"batch_size":args.batch_size,"lr":args.lr,
                    "cut_ratio":args.cut_ratio},
          "pi":{"alpha":args.pi_alpha}}
    with open(os.path.join(args.out_dir,"run_meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2,ensure_ascii=False)


if __name__=="__main__":
    main()
