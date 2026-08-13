# rPPG-PICA-Net

Remote Photoplethysmography (rPPG) system using the **PICA-Net** architecture for heart rate estimation from video footage.

## Overview

This project implements a physics-informed neural network (PICA-Net) for extracting heart rate from remote photoplethysmography (rPPG) videos. The system processes video frames to detect subtle pulsatile motion patterns and predicts cardiac activity using a cross-attention-based architecture with physics-informed temporal modeling.

## Architecture

### PICA-Net (PICA-Net)

The core model consists of:

- **Region Encoders** (`RegionEncoder`): Five spatio-temporal encoders that process appearance and difference frames separately, then fuse them across anatomical regions.
- **Cross-Attention Fusion** (`CrossAttentionFusion`): Learns per-timestep reliability weights across the five region representations.
- **Physics-Informed Temporal Head** (`PhysicsInformedTemporalHead`): Predicts frequency deviation, amplitude, and phase from fused tokens, incorporating explicit phase/frequency/amplitude dynamics inspired by cardiovascular physiology.
- **Output**: Pulse waveform reconstruction, frequency estimates, and amplitude predictions.

### Supporting Components

- **WindowDataset**: Creates sliding-window datasets from video sequences for training/validation.
- **LRUSubjectCache**: Manages subject-level caching of video windows for inference.
- **Pos Algorithm**: Traditional pulse extraction baseline using RGB video analysis.
- **Complexity Analysis**: Computes FLOPs, parameter count, and memory footprint.
- **Statistical Analysis**: Bland-Altman plots, bootstrap confidence intervals, and paired t-tests comparing PICA-Net vs. POS baseline.

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch (CPU or CUDA-enabled GPU)
- NumPy, SciPy, Matplotlib
- Torchvision (for video utilities)

### Installation

```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
python run_pipeline.py \
    --dataset_root ../dataset/UBFC_DATASET/DATASET_2 \
    --out_dir ../final-report/q5_assets \
    --epochs 100
```

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|----------|
| `--dataset_root` | Root directory of the UBFC dataset | `../dataset/UBFC_DATASET/DATASET_2` |
| `--out_dir` | Output directory for results | `../final-report/q5_assets` |
| `--epochs` | Number of training epochs | `15` |
| `--batch_size` | Batch size for training | `8` |
| `--window_frames` | Number of frames per window | `160` |
| `--stride` | Spatial downsampling stride | `80` |
| `--physics_augment` | Enable physics-informed data augmentation | `False` |
| `--device` | Device to use (cpu/mps/cuda) | Auto-detected |
| `--seed` | Random seed for reproducibility | `42` |

### Individual Scripts

- **`retrain_fold3_epochs100.py`**: Retrains the best fold (fold 3) with 100 epochs for comparison against the 50-epoch baseline.
- **`run_cv.py`**: Cross-validation training script.
- **`run_sweep.sh`**: Hyperparameter sweep script.
- **`train.py`**: Standalone training entry point.

## Project Structure

```
rPPG-PICA-Net/
├── README.md
├── complexity.py          # Computational complexity analysis
├── data.py                # Dataset loading and subject splitting
├── evaluate.py            # Metrics computation (MAE, RMSE, Pearson r)
├── lose.py                # Loss functions and training utilities
├── model.py               # PICA-Net model definition
├── pos_baseline.py        # Traditional pulse extraction algorithm
├── run_cv.py              # Cross-validation training pipeline
├── run_pipeline.py        # End-to-end training, evaluation, and reporting
├── run_sweep.sh           # Hyperparameter sweep script
├── statistics/            # Statistical analysis utilities
│   ├── complexity.py
│   └── stats_analysis.py
└── tests/
    └── test_smoke.py      # Smoke tests
```

## Key Outputs

After running `run_pipeline.py`:

- **`fold3_epoch100_summary.json`**: Comprehensive results including per-window predictions, metrics, and plots.
- **`pica_net.pt`**: Trained model checkpoint.
- **`summary.json`**: Summary of split info, results, complexity, and statistical tests.
- **Plots** (saved to `out_dir`):
  - Training/validation loss curves
  - Test-set predictions vs. ground truth (scatter plot)
  - Bland-Altman analysis (PICA-Net vs. POS baseline)

## Evaluation Metrics

- **MAE (Mean Absolute Error)** – Lower is better
- **RMSE (Root Mean Square Error)** – Lower is better
- **Pearson Correlation Coefficient (r)** – Higher is better
- **Bland-Altman Analysis** – Visualizes agreement between PICA-Net and POS baseline
- **Computational Complexity** – Parameter count, FLOPs, memory footprint

## Comparison with POS Baseline

The pipeline compares PICA-Net against a traditional Position-Based (POS) algorithm that extracts pulse from RGB video traces. Typical results show PICA-Net achieving lower MAE and better correlation coefficients, especially on challenging subjects.

## Usage Examples

### Basic Training

```bash
python run_pipeline.py \
    --dataset_root ../dataset/UBFC_DATASET/DATASET_2 \
    --out_dir ./results \
    --epochs 100
```

### Training with Augmentation

```bash
python run_pipeline.py \
    --dataset_root ../dataset/UBFC_DATASET/DATASET_2 \
    --out_dir ./results \
    --epochs 100 \
    --physics_augment \
    --window_frames 160
```

### Inference on a Single Video

```python
from data import find_subjects, WindowDataset
from model import PICANet

# Load subjects
subjects = find_subjects("/path/to/dataset")
print(f"Found {len(subjects)} subjects")

# Create dataset
train_ds = WindowDataset(subjects[:5], cache_dir="/tmp/cache")

# Train model
model = PICANet(out_frames=160, base_freq_hz=72)
model, history = train_model(model, train_ds, None, epochs=50)
```

## License

MIT License - see LICENSE file for details.

## References

- PICA-Net: Physics-Informed Cross-Attention Network for Remote Photoplethysmography
- UBFC Dataset: Universal Benchmark for Face and Body Contour Tracking

## Contributing

Contributions are welcome! Please ensure any changes maintain the experimental design and reproducibility of the rPPG-PICA-Net system.

## Contact

For questions or collaboration, please refer to the project's discussion forum or contact the course organizers.
