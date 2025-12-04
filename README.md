# Rula_NN

This repository contains neural network models and clustering scripts for predictive maintenance and time-series forecasting, specifically focusing on Remaining Useful Life (RUL) estimation using LSTM networks and PCA features.

## Project Structure

### Scripts

*   **`configurableNN.py`**:
    *   **Description**: Pooled LSTM model (PC1-only) with per-part min–max scaling on both PC1 and index. It performs global pooling of data from multiple devices and sorts by scaled index.
    *   **Features**: Configurable LSTM architecture (layers, hidden size), sliding window dataset generation, and plotting of training history and predictions (Fig.15 style).
    *   **Output**: Saves checkpoints (`NN_ckpt.pt`), run metadata (`run_meta.json`), predictions (`predictions_tail_pooled.csv`), and plots to `NN_out/`.

*   **`NN_softpi.py`**:
    *   **Description**: Similar to `configurableNN.py` but introduces a **Soft-PI (Physics-Informed)** penalty (`pi_alpha`) to enforce monotonicity within the prediction horizon (Lp block).
    *   **Features**: Includes an "End of Life" (EoL) threshold detection mechanism and extended free-run predictions.
    *   **Output**: Metrics (`metrics.json`), extended plots, and metadata.

*   **`online_extrapolate_pc1_auto_cfg.py`**:
    *   **Description**: Performs online extrapolation using a trained LSTM model.
    *   **Features**: Automatically infers model configuration (hidden size, layers, etc.) from the checkpoint file or a sidecar JSON. It takes a seed sequence from a CSV file and performs free-run prediction until a failure threshold is reached.
    *   **Usage**: Interactive file selection (via Tkinter) for the model checkpoint and the target device CSV.

*   **`k_means_clustering.py`**:
    *   **Description**: Performs K-means clustering on PCA data (PC1 vs PC2) from the `PCA/` directory.
    *   **Features**: Loads multiple device CSVs, computes progress (0-1), scales features, clusters global data, and generates visualizations (PC1-PC2 clusters, PC1 vs Progress, HI vs Progress).
    *   **Output**: Results saved to `results_kmeans/k{K}/`.

### Directories

*   **`Features/`**: Contains feature CSV files (e.g., `DV_device2_features.csv`).
*   **`Raw_Data/`**: Contains raw data CSV files.
*   **`PCA/`**: Contains PCA score CSV files used as input for clustering.
*   **`NN_out/`**: Default output directory for Neural Network training runs. Contains subdirectories for different experiments (e.g., `LO10LP1`).
*   **`results_kmeans/`**: Output directory for K-means clustering results.

## Installation

Ensure you have Python 3 installed along with the following dependencies:

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

*Note: `tkinter` is also required for the GUI file pickers in some scripts (usually included with standard Python installations).*

## Usage

### 1. Training the Pooled LSTM (`configurableNN.py`)

Run the script with desired parameters. You can specify input files via `--data_csv` or use `--gui_files` to select them interactively.

```bash
# Example: Train with Look-back (Lo)=16, Look-ahead (Lp)=1
python configurableNN.py --data_csv "Features/DV_device2_features.csv,Features/DV_device3_features.csv" --lo 16 --lp 1 --epochs 50 --out_dir NN_out/experiment_1
```

**Key Arguments:**
*   `--lo`: Observation window length (default: 16).
*   `--lp`: Prediction horizon length (default: 1).
*   `--hidden_size`: LSTM hidden units (default: 64).
*   `--num_layers`: Number of LSTM layers (default: 1).
*   `--cut_ratio`: Cut position for training/testing split (default: 0.8).

### 2. Training with Soft-PI (`NN_softpi.py`)

Train with a physics-informed monotonicity penalty.

```bash
# Example: Train with Soft-PI alpha=0.1
python NN_softpi.py --data_csv "Features/DV_device2_features.csv" --pi_alpha 0.1 --out_dir NN_out/softpi_run
```

**Key Arguments:**
*   `--pi_alpha`: Weight for monotonicity penalty (default: 0.0).
*   `--eol_thr`: Threshold for End-of-Life detection (default: 0.98).

### 3. Online Extrapolation (`online_extrapolate_pc1_auto_cfg.py`)

This script typically uses a GUI to select files.

```bash
python online_extrapolate_pc1_auto_cfg.py
```
1.  Select the trained model checkpoint (`.pt`).
2.  Select the CSV file for the device to extrapolate.

### 4. K-Means Clustering (`k_means_clustering.py`)

Cluster PCA data.

```bash
# Example: Run K-means with K=3
python k_means_clustering.py --k 3 --random_state 42
```

**Key Arguments:**
*   `--k`: Number of clusters (default: 3).
*   `--pca_dir`: Directory containing PCA CSVs (default: `./PCA`).

## Configuration

*   **Command Line Arguments**: Most scripts use `argparse` for configuration. Use `python <script.py> --help` to see all available options.
*   **Metadata**: Training runs save a `run_meta.json` file containing the configuration used, scaling parameters, and file lists.
*   **Inference Config**: `online_extrapolate_pc1_auto_cfg.py` attempts to read a sidecar JSON file (e.g., `model.json` next to `model.pt`) or infers parameters from the model's state dictionary.
