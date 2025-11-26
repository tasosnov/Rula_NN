# -*- coding: utf-8 -*-
"""
Online Extrapolation για PC1-only LSTM (Lo = SEQ_LEN)
- Φορτώνει εκπαιδευμένο μοντέλο (.pt) με **αυτόματο** έλεγχο consistency:
  * Προσπαθεί να διαβάσει hidden_size / num_layers / bidirectional / input_size
    από sidecar JSON (ίδιο base-name με .json) ή/και από το state_dict.
- Επιλέγεις ΕΝΑ CSV device
- Per-device FULL min–max στον PC1 (ώστε thr=1.0 να σημαίνει 'max του αρχείου')
- Seed: πρώτα N_SEED σημεία (π.χ. N_SEED = 2*Lo)
- Free-run μέχρι να φτάσει thr ή μέχρι INFER_MAX_STEPS
- Αποθήκευση: plot + CSVs στο OUT_DIR

Απαιτήσεις:
    pip install torch numpy pandas matplotlib
"""

import os, os.path as osp, json, math, re
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox

# =============== ΡΥΘΜΙΣΕΙΣ (προσαρμόζεις) ===============
OUT_DIR         = "./out_online_extrapolation"
SEQ_LEN         = 12           # Lo: πρέπει να ταιριάζει με το training window length
INFER_N_SEED    = 3 * SEQ_LEN  # πόσα πρώτα σημεία παίρνεις ως seed
PC1_FAIL_THR    = 1.0          # στο per-device full min–max, 1.0 = 'max του αρχείου'
INFER_MAX_STEPS = 5000         # cap βημάτων για free-run
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)
# ==========================================================

# ========================= Model ==========================
class LSTMForecaster(nn.Module):
    """
    LSTM forecaster με παραμέτρους που προσαρμόζονται στο checkpoint.
    - input_size: default 1 (PC1-only), αλλά μπορεί να ανιχνευθεί από το state_dict.
    - hidden_size, num_layers, bidirectional: αντλούνται από το state_dict / JSON.
    - fc input dim = hidden_size * (2 if bidirectional else 1)
    """
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc   = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)     # (B,L,H*D)
        out = out[:, -1, :]       # (B,H*D)
        return self.fc(out)       # (B,1)


# ======================= Helpers ==========================
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
        # Αν η πρώτη στήλη είναι numeric, τη θεωρούμε epoch
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
    """Απλός min–max scaler: fit στο πλήρες range."""
    def __init__(self, a=0.0, b=1.0):
        self.a=a; self.b=b; self.min_=None; self.max_=None
    def fit(self, x):
        x = np.asarray(x, float)
        self.min_ = float(np.min(x)); self.max_ = float(np.max(x))
        if math.isclose(self.min_, self.max_):
            self.max_ = self.min_ + 1.0
        return self
    def transform(self, x):
        x = np.asarray(x, float)
        return self.a + (self.b-self.a) * ((x - self.min_) / (self.max_ - self.min_))


# ============== Checkpoint reading & config inference ==============
def try_read_sidecar_json(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Αν υπάρχει file με ίδιο base-name και κατάληξη .json, το φορτώνει.
    Περιμένουμε κλειδιά: input_size, hidden_size, num_layers, bidirectional.
    """
    base, _ = osp.splitext(model_path)
    json_path = base + ".json"
    if osp.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Ελάχιστος έλεγχος ορθότητας
            keys = {"input_size", "hidden_size", "num_layers", "bidirectional"}
            if any(k in cfg for k in keys):
                return cfg
        except Exception:
            pass
    return None


def infer_lstm_cfg_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Εξάγει input_size, hidden_size, num_layers, bidirectional από state_dict.
    - Βασιζόμαστε στα weight_ih_l{layer} (σχήμα: [4H, I]) και weight_hh_l{layer} ([4H, H]).
    - Αν υπάρχουν keys με suffix "_reverse", το θεωρούμε bidirectional.
    """
    # Συγκεντρώνουμε όλα τα layers που υπάρχουν
    layer_pat = re.compile(r"^lstm\.weight_ih_l(\d+)(?:_reverse)?$")
    layers = []
    bidirectional = False
    input_size = None
    hidden_size = None

    for k, v in state.items():
        m = layer_pat.match(k)
        if m:
            l = int(m.group(1))
            layers.append(l)
            if "reverse" in k:
                bidirectional = True
            # v shape: (4H, I)
            if v.ndim == 2:
                fourH, I = v.shape
                if input_size is None:
                    input_size = int(I)
                if hidden_size is None:
                    # H must be fourH // 4
                    hidden_size = int(fourH // 4)

    num_layers = (max(layers) + 1) if layers else 1

    # Fallbacks αν δεν βρέθηκαν ρητά
    if hidden_size is None:
        # Δοκιμή από weight_hh_l0
        w_hh = state.get("lstm.weight_hh_l0", None)
        if w_hh is not None and w_hh.ndim == 2:
            hidden_size = int(w_hh.shape[1])
        else:
            hidden_size = 32

    if input_size is None:
        w_ih = state.get("lstm.weight_ih_l0", None)
        if w_ih is not None and w_ih.ndim == 2:
            input_size = int(w_ih.shape[1])
        else:
            input_size = 1

    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional
    }


def load_checkpoint(model_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Φορτώνει checkpoint (.pt) και επιστρέφει (state_dict, raw_object).
    Υποστηρίζει:
    - Καθαρό state_dict
    - Checkpoint dict που περιέχει 'state_dict' ή 'model_state'
    """
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict):
        # διάφορες πιθανές συμβάσεις
        for key in ("state_dict", "model_state", "model", "net", "network"):
            if key in obj and isinstance(obj[key], dict):
                return {"state_dict": obj[key], "raw": obj}
        # Ίσως είναι απευθείας state_dict
        tensor_like = [k for k, v in obj.items() if torch.is_tensor(v)]
        if tensor_like:
            return {"state_dict": obj, "raw": obj}
    elif isinstance(obj, nn.Module):
        return {"state_dict": obj.state_dict(), "raw": obj}

    raise RuntimeError("Δεν αναγνωρίζω format μοντέλου. Δώσε .pt με state_dict ή dict['state_dict'].")


def build_model_from_checkpoint(model_path: str, device: torch.device) -> nn.Module:
    """
    - Προσπαθεί να διαβάσει config από sidecar JSON.
    - Αλλιώς, εξάγει config από το state_dict.
    - Φτιάχνει LSTMForecaster με τις σωστές διαστάσεις και φορτώνει state.
    """
    ck = load_checkpoint(model_path, device)
    state = ck["state_dict"]

    # 1) Προτίμηση σε sidecar JSON
    cfg = try_read_sidecar_json(model_path)

    # 2) Διαφορετικά, inference από state_dict
    if not cfg:
        cfg = infer_lstm_cfg_from_state(state)

    # Έλεγχοι συμβατότητας
    if cfg["input_size"] != 1:
        # Θα δουλέψει ΜΟΝΟ αν το input σου είναι μονοδιάστατο (PC1).
        # Αν είναι αλλιώς, πρέπει να αλλάξεις και την προετοιμασία εισόδου.
        print(f"[WARN] input_size={cfg['input_size']} στο checkpoint, αλλά το script τροφοδοτεί 1 feature (PC1).")

    model = LSTMForecaster(
        input_size=int(cfg.get("input_size", 1)),
        hidden_size=int(cfg.get("hidden_size", 32)),
        num_layers=int(cfg.get("num_layers", 1)),
        bidirectional=bool(cfg.get("bidirectional", False)),
    ).to(device)

    # Προσπάθησε να φορτώσεις απευθείας
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # Μήπως τα keys έχουν prefix (π.χ. "module.") ή άλλο name; δοκίμασε απλό remap.
        print("[INFO] Αποτυχία άμεσης φόρτωσης state_dict, δοκιμή remap keys…")
        new_state = {}
        for k, v in state.items():
            nk = k
            nk = nk.replace("model.", "")
            nk = nk.replace("module.", "")
            new_state[nk] = v
        model.load_state_dict(new_state)
    return model


# ================== Inference core (όπως πριν) ==================
def infer_free_run_until_threshold(seed_scaled: np.ndarray,
                                   model: nn.Module,
                                   device: torch.device,
                                   seq_len: int,
                                   thr: float,
                                   max_steps: int) -> np.ndarray:
    """
    Free-run ξεκινώντας από seed_scaled (len >= seq_len) μέχρι να φτάσει/ξεπεράσει thr
    ή μέχρι max_steps. Επιστρέφει ΜΟΝΟ τα new predicted points (όχι τα seed).
    """
    assert len(seed_scaled) >= seq_len, "Seed length must be >= seq_len."
    win = seed_scaled[-seq_len:].astype(np.float32).copy()
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            X = torch.from_numpy(win.reshape(1, seq_len, 1)).float().to(device)
            yhat = model(X).cpu().numpy().ravel()[0]
            out.append(yhat)
            if yhat >= thr:
                break
            # clamp μικρο-σταθερότητας (προαιρετικό)
            if not np.isfinite(yhat):
                break
            win = np.concatenate([win[1:], [float(yhat)]])  # cast σε float για ασφάλεια
    return np.array(out, dtype=np.float32)


def find_crossing_index(series: np.ndarray, thr: float) -> Optional[int]:
    """Πρώτο index i όπου series[i] >= thr. Αν δεν υπάρχει, επιστρέφει None."""
    idx = np.where(series >= thr)[0]
    return int(idx[0]) if idx.size else None


def build_x_pred(x_full: np.ndarray, start_idx: int, steps: int) -> np.ndarray:
    """
    Δημιουργεί x για τα predictive σημεία:
    - Αν χωράνε μέσα στο full, τα παίρνει από εκεί.
    - Αν περισσεύουν, επεκτείνει γραμμικά πέρα από το τέλος με ίδιο dx.
    """
    N = len(x_full)
    dx = (x_full[1]-x_full[0]) if N>1 else 1.0
    remain = max(0, N - (start_idx+1))
    take = min(steps, remain)
    xs = []
    if take > 0:
        xs.append(x_full[start_idx+1:start_idx+1+take])
    extra = steps - take
    if extra > 0:
        last = x_full[-1] if N>0 else 0.0
        xs.append(last + dx * np.arange(1, extra+1))
    return np.concatenate(xs) if xs else np.array([], dtype=float)


# =========================== Main ==========================
def main():
    # 1) Διαλόγους επιλογής αρχείων (μοντέλο + device CSV)
    root = tk.Tk(); root.withdraw()
    model_path = filedialog.askopenfilename(
        title="Διάλεξε trained μοντέλο (.pt)",
        filetypes=[("PyTorch model","*.pt"), ("All files","*.*")]
    )
    if not model_path:
        messagebox.showerror("Ακύρωση", "Δεν επέλεξες μοντέλο."); return

    csv_path = filedialog.askopenfilename(
        title="Διάλεξε ΕΝΑ CSV για online extrapolation",
        filetypes=[("CSV","*.csv"), ("All files","*.*")]
    )
    if not csv_path:
        messagebox.showerror("Ακύρωση", "Δεν επέλεξες CSV."); return

    # 2) Φόρτωμα μοντέλου με αυτόματο inference παραμέτρων
    try:
        model = build_model_from_checkpoint(model_path, DEVICE)
    except Exception as e:
        messagebox.showerror("Σφάλμα φόρτωσης μοντέλου", str(e))
        return

    # 3) Φόρτωμα & καθάρισμα CSV
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception as e:
        messagebox.showerror("Σφάλμα CSV", f"Αποτυχία ανάγνωσης CSV: {e}")
        return

    ecol, pcol, df = detect_columns(df)
    d = df[[ecol, pcol]].copy()
    d.columns = ["epoch", "pc1"]
    d = d.dropna().sort_values("epoch").drop_duplicates("epoch").reset_index(drop=True)
    d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce")
    d = d.dropna().reset_index(drop=True)

    if len(d) < max(SEQ_LEN, INFER_N_SEED):
        messagebox.showerror("Σφάλμα", f"Λίγα δείγματα ({len(d)}) για SEQ_LEN={SEQ_LEN} και N_SEED={INFER_N_SEED}.")
        return

    # 4) Per-device FULL min–max στον PC1 και επίσης στο epoch (για x-άξονα)
    sc_pc1 = MinMax1D(0.0, 1.0).fit(d["pc1"].values)
    sc_ep  = MinMax1D(0.0, 1.0).fit(d["epoch"].values)

    pc1_scaled_full = sc_pc1.transform(d["pc1"].values)     # (N,)
    e_scaled_full   = sc_ep.transform(d["epoch"].values)     # (N,)

    # Seed
    n_seed = min(INFER_N_SEED, len(pc1_scaled_full))
    seed   = pc1_scaled_full[:n_seed].copy()

    # 5) Free-run μέχρι thr ή max_steps
    pred_traj = infer_free_run_until_threshold(
        seed_scaled=seed,
        model=model,
        device=DEVICE,
        seq_len=SEQ_LEN,
        thr=PC1_FAIL_THR,
        max_steps=INFER_MAX_STEPS
    )

    # Πού περνάει το threshold;
    cross_idx_rel   = find_crossing_index(pred_traj, PC1_FAIL_THR)   # index εντός pred_traj
    cross_idx_total = (n_seed + cross_idx_rel) if cross_idx_rel is not None else None

    # x-άξονας (scaled epoch) για seed και pred
    x_seed = e_scaled_full[:n_seed]
    x_pred = build_x_pred(e_scaled_full, start_idx=n_seed-1, steps=len(pred_traj))

    # Real epoch mapping για τα predicted (αν θες raw epoch info)
    epoch_pred = None
    if len(x_pred) > 0:
        epoch_pred = sc_ep.min_ + x_pred * (sc_ep.max_ - sc_ep.min_)

    # 6) Plot
    base = osp.splitext(osp.basename(csv_path))[0]
    plt.figure(figsize=(9,5))
    plt.scatter(x_seed, seed, s=14, alpha=0.85, label=f"Seed (first {n_seed})")
    if len(pred_traj) > 0:
        plt.scatter(x_pred, pred_traj, s=12, c="black", alpha=0.9, label="Predictive (free-run)")
    plt.axhline(PC1_FAIL_THR, ls="--", c="red", lw=1.2, label=f"Threshold = {PC1_FAIL_THR:.2f}")

    # Μαρκάρισμα crossing
    if cross_idx_total is not None:
        if cross_idx_total < n_seed:
            x_cross = x_seed[cross_idx_total]
            y_cross = seed[cross_idx_total]
        else:
            j = cross_idx_total - n_seed
            if j < len(x_pred):
                x_cross = x_pred[j]; y_cross = pred_traj[j]
            else:
                x_cross = None; y_cross = None
        if x_cross is not None:
            plt.scatter([x_cross], [y_cross], s=40, c="red", zorder=5, label="First crossing")

    plt.xlabel("Scaled epoch (per device)")
    plt.ylabel("PC1 (scaled per device)")
    plt.title(f"Online Extrapolation — seed={n_seed}, Lo={SEQ_LEN} — {base}")
    plt.legend(loc="best"); plt.tight_layout()
    fig_path = osp.join(OUT_DIR, f"online_extrapolation_{base}.png")
    plt.savefig(fig_path, dpi=160); plt.close()

    # 7) CSVs
    out_seed = pd.DataFrame({
        "seed_idx": np.arange(n_seed),
        "seed_pc1_scaled": seed,
        "seed_epoch": d["epoch"].values[:n_seed]
    })
    out_seed.to_csv(osp.join(OUT_DIR, f"online_seed_{base}.csv"), index=False)

    out_pred = pd.DataFrame({
        "pred_step": np.arange(1, len(pred_traj)+1),
        "pred_pc1_scaled": pred_traj,
        "pred_epoch": epoch_pred if epoch_pred is not None else np.nan
    })
    out_pred.to_csv(osp.join(OUT_DIR, f"online_pred_{base}.csv"), index=False)

    # 8) Περίληψη στην κονσόλα
    print(f"[OK] Figure: {osp.abspath(fig_path)}")
    if cross_idx_total is not None:
        if cross_idx_total < n_seed:
            print(f"[INFO] Threshold reached μέσα στο seed (index {cross_idx_total}).")
        else:
            steps_to_thr = cross_idx_total - n_seed + 1
            print(f"[INFO] Predicted steps-to-threshold from seed: {steps_to_thr}")
            if epoch_pred is not None and (steps_to_thr-1) < len(epoch_pred):
                print(f"[INFO] Predicted epoch at threshold (approx): {epoch_pred[steps_to_thr-1]:.3f}")
    else:
        print(f"[INFO] Threshold ΔΕΝ επιτεύχθηκε σε {len(pred_traj)} predicted steps (cap={INFER_MAX_STEPS}).")


if __name__ == "__main__":
    main()
