# -*- coding: utf-8 -*-
"""
UNIFIED LSTM (PC1-only) με:
- Per-part FULL min–max για PC1 (fit στο πλήρες αρχείο κάθε part)
- Per-part min–max για epoch ΜΟΝΟ για ταξινόμηση/x-άξονα
- Συγχώνευση όλων των parts σε μία ενιαία σειρά βάσει scaled-epoch (0→αρχή part, 1→τέλος part)
- LSTM: Lo=9, input=PC1′ (Lp=1), στόχος=PC1′(t+1)
- Ρυθμιζόμενος cut (μέχρι που εκπαιδεύω) & PRED_STEPS (πόσα free-run)
- Early Stop διακόπτης (on/off)
- Ενιαίο plot: Actual / Training (ως cut) / Predictive

Outputs → ./out_unified_pc1_only_cutpred
"""

import os, json, math
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

import tkinter as tk
from tkinter import filedialog, messagebox

# =========================== ΡΥΘΜΙΣΕΙΣ ===========================
OUT_DIR      = "./out_unified_pc1_only_cutpred"
SEQ_LEN      = 12                # Lo
EPOCHS       = 40
BATCH_SIZE   = 128
LR           = 2e-3
HIDDEN_UNITS = 64
SEED         = 42

# ---- Ορισμός cut (μέχρι πού "βλέπουμε") ----
# Επιλογές:
#   "ratio":   χρησιμοποιεί TRAIN_RATIO και VAL_RATIO μέσα στο observed τμήμα
#   "index":   απόλυτος δείκτης TRAIN_END_INDEX (0..N-1)
#   "last_k":  κρατά τα τελευταία LAST_K_FOR_LATER δείγματα για "μετά"
CUT_MODE          = "ratio"     # "ratio" | "index" | "last_k"
TRAIN_RATIO       = 0.70        # όταν CUT_MODE="ratio"
VAL_RATIO         = 0.15
TRAIN_END_INDEX   = None        # π.χ. 1200 όταν CUT_MODE="index"
LAST_K_FOR_LATER  = 200         # όταν CUT_MODE="last_k"

# ---- Extrapolation ----
# "to_end": free-run μέχρι το τέλος των υπαρχόντων δειγμάτων
# ακέραιος: π.χ. 80 → free-run 80 βήματα (συνεχίζει και πέρα από το τέλος)
PRED_STEPS        = "to_end"

# ---- Early Stopping ----
EARLY_STOP = False
PATIENCE   = 8
# ================================================================

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)

# ======================== ΒΟΗΘΗΤΙΚΑ ============================
def detect_columns(df: pd.DataFrame) -> Tuple[str, str, pd.DataFrame]:
    """Ανίχνευση epoch/idx & PC1 με fallbacks."""
    epoch_candidates = ["epoch","epochs","idx","index","time","t"]
    pc1_candidates   = ["pc1","pc_1","pc1_score","pca1","pc1(t)","pc1_value"]

    e = next((c for c in df.columns if c.lower() in epoch_candidates), None)
    p = None
    for c in df.columns:
        if any(k in c.lower() for k in pc1_candidates):
            p = c; break

    if e is None:
        if np.issubdtype(df.iloc[:,0].dtype, np.number):
            e = df.columns[0]
        else:
            df = df.reset_index(drop=True)
            df.insert(0, "epoch_idx", np.arange(len(df)))
            e = "epoch_idx"

    if p is None:
        for c in df.columns:
            if c != e and np.issubdtype(df[c].dtype, np.number):
                p = c; break
    if p is None:
        raise ValueError("PC1 column not found.")
    return e, p, df

class MinMax1D:
    """Απλός 1D min–max (fit στο full range)."""
    def __init__(self, a=0.0, b=1.0):
        self.a=a; self.b=b; self.min_=None; self.max_=None
    def fit(self, x):
        x = np.asarray(x, float)
        self.min_ = float(np.min(x)); self.max_ = float(np.max(x))
        if math.isclose(self.min_, self.max_): self.max_ = self.min_ + 1.0
        return self
    def transform(self, x):
        x = np.asarray(x, float)
        return self.a + (self.b-self.a)*((x - self.min_) / (self.max_-self.min_))

def make_windows(series: np.ndarray, a: int, b: int, L: int):
    """Windows Lo=L, στόχος το επόμενο δείγμα (t+1). Επιστρέφει X:(?,L,1), y:(?,)."""
    Xs, ys = [], []
    for t in range(a + L - 1, b - 1):
        Xs.append(series[t-L+1:t+1].reshape(L,1))
        ys.append(series[t+1])
    X = np.array(Xs, np.float32) if Xs else np.zeros((0, L, 1), np.float32)
    y = np.array(ys, np.float32) if ys else np.zeros((0,), np.float32)
    return X, y

def compute_bounds(N: int):
    """
    Επιστρέφει (train_end_excl, val_end_excl, cut_t) για την ΕΝΙΑΙΑ σειρά μήκους N.
    cut_t = τελευταίο observed index πριν αρχίσει το free-run.
    """
    if CUT_MODE == "ratio":
        n_tr = max(SEQ_LEN, int(N*TRAIN_RATIO))
        n_va = max(0, int(N*VAL_RATIO))
        train_end_excl = n_tr
        val_end_excl   = min(N, n_tr + n_va)
        cut_t = max(SEQ_LEN-1, (val_end_excl - 1 if n_va>0 else n_tr - 1))
        return train_end_excl, val_end_excl, cut_t

    if CUT_MODE == "index":
        if TRAIN_END_INDEX is None or not (SEQ_LEN <= TRAIN_END_INDEX < N):
            raise ValueError("Δώσε TRAIN_END_INDEX στο [SEQ_LEN, N-1).")
        train_end_excl = TRAIN_END_INDEX + 1
        n_va = max(0, int(N*VAL_RATIO))
        val_end_excl = min(N, train_end_excl + n_va)
        cut_t = max(SEQ_LEN-1, (val_end_excl - 1 if n_va>0 else train_end_excl - 1))
        return train_end_excl, val_end_excl, cut_t

    if CUT_MODE == "last_k":
        K = max(0, min(int(LAST_K_FOR_LATER), N-SEQ_LEN-1))
        val_end_excl = N - K            # observed τμήμα
        n_obs = val_end_excl
        train_end_excl = max(SEQ_LEN, int(n_obs * TRAIN_RATIO))
        cut_t = max(SEQ_LEN-1, val_end_excl - 1)
        return train_end_excl, val_end_excl, cut_t

    raise ValueError("CUT_MODE must be 'ratio', 'index', or 'last_k'.")

# =================== Torch dataset & model ===================
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return self.X[i], self.y[i]

class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(hidden,1)
    def forward(self, x):
        out,_ = self.lstm(x); out = out[:,-1,:]
        return self.fc(out)

def evaluate(model, loader, device, desc=None):
    model.eval(); losses=[]; crit=nn.MSELoss()
    it = loader if desc is None else tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for Xb,yb in it:
            Xb=Xb.to(device); yb=yb.to(device)
            losses.append(crit(model(Xb), yb).item())
    return float(np.mean(losses)) if losses else float("nan")

# ===== TF (μέχρι cut) & Free-run =====
def tf_curve_scaled(series: np.ndarray, model, device, L: int, up_to_idx: int) -> np.ndarray:
    """Teacher-forcing 1-step ΜΟΝΟ μέχρι up_to_idx. Επιστρέφει NaN για τα υπόλοιπα."""
    N=len(series); pred=np.full(N, np.nan, np.float32)
    with torch.no_grad():
        end = min(up_to_idx, N-2)
        for t in range(L-1, end+1):
            X = series[t-L+1:t+1].reshape(1,L,1)
            yhat = model(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()[0]
            pred[t+1]=yhat
    return pred

def free_run_scaled(series: np.ndarray, model, device, start_t: int, steps: int, L: int) -> np.ndarray:
    """Free-run μετά το cut. Αν steps > διαθέσιμα, συνεχίζει πέρα από το τέλος."""
    win = series[start_t-L+1:start_t+1].astype(np.float32).copy()
    out=[]
    with torch.no_grad():
        for _ in range(steps):
            X = win.reshape(1,L,1)
            yhat = model(torch.from_numpy(X).float().to(device)).cpu().numpy().ravel()[0]
            out.append(yhat)
            win = np.concatenate([win[1:], [yhat]])
    return np.array(out, np.float32)

# =============================== MAIN ===============================
def main():
    # 1) Επιλογή αρχείων
    root = tk.Tk(); root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Διάλεξε CSV parts (πολλαπλή επιλογή)",
        filetypes=[("CSV","*.csv"),("All files","*.*")]
    )
    if not paths:
        messagebox.showerror("Ακύρωση", "Δεν επιλέχθηκαν αρχεία."); return
    paths = list(paths)

    # 2) Φόρτωμα & per-part FULL min–max
    pooled = []
    for fp in paths:
        df = pd.read_csv(fp, sep=None, engine="python")
        ecol, pcol, df = detect_columns(df)
        d = df[[ecol, pcol]].copy()
        d.columns = ["epoch","pc1"]
        d = d.dropna().sort_values("epoch").drop_duplicates("epoch").reset_index(drop=True)
        d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce")
        d = d.dropna().reset_index(drop=True)
        if len(d) < SEQ_LEN + 2:
            print(f"[WARN] Skip {os.path.basename(fp)} (n={len(d)})")
            continue

        sc_pc1  = MinMax1D().fit(d["pc1"].values)     # FULL range για PC1
        sc_ep   = MinMax1D().fit(d["epoch"].values)   # FULL range για epoch (για ταξινόμηση/x-άξονα)

        pc1s = sc_pc1.transform(d["pc1"].values)
        es   = sc_ep.transform(d["epoch"].values)

        pooled.append(pd.DataFrame({"e_scaled": es, "pc1_scaled": pc1s}))

    if not pooled:
        messagebox.showerror("Σφάλμα", "Κανένα αρχείο δεν είχε επαρκή δείγματα."); return

    # 3) Ενιαία σειρά
    U = pd.concat(pooled, ignore_index=True).sort_values("e_scaled").reset_index(drop=True)
    series = U["pc1_scaled"].values.astype(np.float32)
    N = len(series)

    # 4) Υπολογισμός ορίων training/validation/cut
    train_end_excl, val_end_excl, cut_t = compute_bounds(N)
    # guards
    train_end_excl = max(SEQ_LEN, min(train_end_excl, N))
    val_end_excl   = max(train_end_excl, min(val_end_excl, N))
    cut_t          = max(SEQ_LEN-1, min(cut_t, N-2))

    # 5) Windows
    Xtr, ytr = make_windows(series, 0,             train_end_excl, SEQ_LEN)
    Xva, yva = make_windows(series, train_end_excl,val_end_excl,   SEQ_LEN)
    Xte, yte = make_windows(series, val_end_excl,  N,              SEQ_LEN)

    with open(os.path.join(OUT_DIR,"dataset_shapes.json"),"w",encoding="utf-8") as f:
        json.dump({
            "N":N,
            "train_end_excl":int(train_end_excl),
            "val_end_excl":int(val_end_excl),
            "cut_t":int(cut_t),
            "train":list(Xtr.shape),
            "val":list(Xva.shape),
            "test":list(Xte.shape)
        }, f, indent=2)

    # 6) DataLoaders
    train_loader = DataLoader(SeqDS(Xtr,ytr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SeqDS(Xva,yva), batch_size=BATCH_SIZE, shuffle=False) if len(Xva) else None
    test_loader  = DataLoader(SeqDS(Xte,yte), batch_size=BATCH_SIZE, shuffle=False) if len(Xte) else None

    # 7) Μοντέλο / Εκπαίδευση
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMForecaster(hidden=HIDDEN_UNITS).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    crit   = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    tr_hist, va_hist = [], []
    best_path = os.path.join(OUT_DIR, "best_model.pt")

    for ep in trange(1, EPOCHS+1, desc="Epochs", leave=True):
        model.train(); running=0.0
        bar = tqdm(train_loader, desc=f"Train ep {ep}", leave=False)
        for i,(Xb,yb) in enumerate(bar, start=1):
            Xb=Xb.to(device); yb=yb.to(device)
            opt.zero_grad(); pred=model(Xb); loss=crit(pred,yb)
            loss.backward(); opt.step()
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.6f}", avg=f"{running/i:.6f}")

        tr_mse = evaluate(model, train_loader, device, desc="Eval train")
        va_mse = evaluate(model, val_loader,   device, desc="Eval val") if val_loader is not None else float("nan")
        tr_hist.append(tr_mse); va_hist.append(va_mse)
        tqdm.write(f"[Epoch {ep}/{EPOCHS}] Train MSE: {tr_mse:.6f}" + (f" | Val MSE: {va_mse:.6f}" if not math.isnan(va_mse) else " | Val: ---"))

        improved = (not math.isnan(va_mse)) and (va_mse < best_val - 1e-7)
        if improved:
            best_val = va_mse
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if EARLY_STOP and (val_loader is not None) and (wait >= PATIENCE):
                tqdm.write("Early stopping: patience reached.")
                break

    # αν δεν υπήρξε val ή δεν υπήρξε βελτίωση, σώσε το τρέχον ως "best"
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
    # και σίγουρα σώσε και το τελικό
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "final_model.pt"))

    # Ιστορικό
    plt.figure()
    plt.plot(tr_hist, label="train")
    if not all(math.isnan(v) for v in va_hist):
        plt.plot([v for v in va_hist], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
    plt.title("Training History (Unified PC1-only)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"training_history.png")); plt.close()

    # Test MSE (αν υπάρχει)
    test_mse = evaluate(model, test_loader, device, desc="Eval test") if test_loader is not None else float("nan")
    with open(os.path.join(OUT_DIR,"test_metrics_scaled.json"),"w") as f:
        json.dump({"mse_scaled":None if math.isnan(test_mse) else test_mse}, f, indent=2)

    # 8) TF (μέχρι cut) & Free-run
    model.load_state_dict(torch.load(best_path, map_location=device))
    series_np = series  # (N,)
    tf_scaled = tf_curve_scaled(series_np, model, device, SEQ_LEN, up_to_idx=cut_t)

    # Πόσα βήματα free-run;
    if PRED_STEPS == "to_end":
        steps = (N - 1) - cut_t
    else:
        steps = max(0, int(PRED_STEPS))
    pred_s = free_run_scaled(series_np, model, device, cut_t, steps, SEQ_LEN) if steps>0 else np.array([])

    # x-άξονας ενιαίος (0..1)
    x_full = np.linspace(0.0, 1.0, num=N, endpoint=True)
    # για προεκτάσεις πέρα από το τέλος, κρατάμε ίδιο βήμα στον x
    dx = (x_full[1]-x_full[0]) if N>1 else 1.0
    x_pred = x_full[cut_t+1:cut_t+1+len(pred_s)] if steps!="to_end" else np.linspace(x_full[cut_t+1], x_full[-1], num=len(pred_s), endpoint=True)
    if len(pred_s)>0 and len(x_pred)==0:
        # αν κάνουμε επιπλέον πέρα απ' το τέλος, συνεχίζουμε γραμμικά
        last = x_full[-1] if N>0 else 0.0
        x_pred = last + dx * np.arange(1, len(pred_s)+1)

    # 9) Ενιαίο plot
    plt.figure(figsize=(9.2,5.2), dpi=130)
    plt.scatter(x_full, series_np, s=9, alpha=0.5, label="Actual (scaled)", c="#1f77b4")
    valid = ~np.isnan(tf_scaled)
    plt.plot(x_full[valid], tf_scaled[valid], linewidth=1.8, label="Training (scaled, up to cut)", c="#d62728")
    if len(pred_s)>0:
        plt.scatter(x_pred, pred_s, s=12, alpha=0.9, label=f"Predictive (scaled)", c="black")
    plt.axvline(x_full[cut_t], ls="--", lw=1.0, c="gray", alpha=0.8, label="cut")

    plt.xlabel("Unified normalized position (0–1, sorted by per-part scaled epoch)")
    plt.ylabel("PC1 (scaled 0–1 per part)")
    plt.title("UNIFIED — Actual vs Training (to cut) vs Predictive (free-run)")
    plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"unified_fig15_scaled_cutpred.png"), dpi=160)
    plt.close()

    print(f"OK. Outputs → {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
