# NC-Expert

Official implementation of **"NC-Expert: A Node-Clustering-Guided Dual-Channel Heterogeneous Expert Network for Traffic Flow Prediction"**.

<!-- TODO: replace the placeholders below after the paper is accepted -->
<!-- [![Paper](https://img.shields.io/badge/paper-XXXXX-blue)]() [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

NC-Expert predicts future traffic flow from node-wise heterogeneous sensor networks in two stages:

1. **MANC** — a masked-autoencoding-based node clustering module. A hierarchical CNN–GRU encoder learns node representations by reconstructing randomly masked observations; K-means (K = 2) then partitions nodes into a **high-flow dynamic** subset and a **low-flow stable** subset.
2. **DCHE** — a dual-channel heterogeneous expert module. The two node subsets are routed to two structurally different experts:
   - **high-flow dynamic expert**: dynamic graph built from a *node memory-aware* adjacency **and** a *feature self-correlation* adjacency, aggregated by diffusion convolution;
   - **low-flow stable expert**: dynamic graph built from the *node memory-aware* adjacency only, for a lighter and more stable spatial model.

Both experts share the same temporal architecture (day-of-week / time-of-day embeddings + Transformer) but keep independent parameters. Predictions are placed back into the original node order to form the final output.

```
X (T×N) ─► random mask ─► CNN (L=4) ─► GRU (M=1) ─► FC ─► Z (N×D)
                                                        │
                                                  K-means (K=2)
                                                        │
                        ┌───────────────────────────────┴───────────────────────────────┐
                  high-flow subset                                          low-flow subset
                  DynamicExpert                                             StableExpert
            (memory adj + self-corr adj)                                  (memory adj only)
              DGCN ─► Transformer                                        DGCN ─► Transformer
                        └───────────────────────────────┬───────────────────────────────┘
                                          inverse routing ─► Ŷ (H×N)
```

---

## Repository structure

```
NC-Expert/
├── configs/                     # per-dataset hyperparameters
│   ├── PeMS03.yaml
│   ├── PeMS04.yaml
│   ├── PeMS07.yaml
│   └── PeMS08.yaml
├── data/                        # place downloaded datasets here (not tracked)
├── models/
│   ├── manc.py                  # masking, CNN-GRU encoder, MLP decoder
│   ├── dche.py                  # routing + dual-channel fusion
│   └── experts/
│       ├── dynamic_expert.py    # high-flow dynamic expert
│       └── stable_expert.py     # low-flow stable expert
├── scripts/
│   ├── prepare_data.py          # raw PeMS -> flow.npz
│   ├── pretrain_manc.py         # stage 1: masked reconstruction + K-means
│   ├── train.py                 # stage 2: dual-channel expert training
│   ├── evaluate.py              # MAE / RMSE / MAPE over 12 horizons
│   ├── run_ablation.py          # Table 5 configurations
│   ├── run_robustness.py        # random / consecutive missing data
│   └── reproduce_all.sh         # run everything end to end
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Environment

Tested with Python 3.8+, PyTorch 1.10+, CUDA 11.x on a single NVIDIA GPU (24 GB is sufficient for all four datasets).

```bash
conda create -n ncexpert python=3.9 -y
conda activate ncexpert
pip install -r requirements.txt
# requirements.txt: torch>=1.10, numpy, pandas, scikit-learn, pyyaml, tqdm
```

The model is deterministic given a fixed seed; all reported numbers use five runs with different seeds.

---

## Data preparation

Download the PeMS03 / PeMS04 / PeMS07 / PeMS08 traffic-flow files (Caltrans PeMS, 5-minute interval) from the public links released with Graph WaveNet / ASTGNN, then convert them:

```bash
python scripts/prepare_data.py --raw_dir /path/to/raw/PeMS04 --out_dir data/PeMS04
```

Expected layout after conversion:

```
data/PeMS04/
└── flow.npz        # keys: "flow" -> float32 array of shape [T, N]
```

Notes:

- Only the **traffic flow** channel is used, as in the paper.
- No static adjacency matrix is required: both experts learn their graphs dynamically.
- Chronological split is **6 : 2 : 2** (train / val / test) and is handled inside `scripts/train.py`.
- For the missing-data experiments, the *test-time inputs* are perturbed (zeros), while the training data stay intact.

Dataset statistics used in the paper:

| Dataset | #Nodes | Period | #Steps T |
|---|---|---|---|
| PeMS03 | 358 | 2018-09-01 – 2018-11-30 | 26,208 |
| PeMS04 | 307 | 2018-01-01 – 2018-02-28 | 16,992 |
| PeMS07 | 883 | 2017-05-01 – 2017-08-31 | 28,224 |
| PeMS08 | 170 | 2016-07-01 – 2016-08-31 | 17,856 |

---

## Quick start

```bash
# Stage 1: masked-autoencoding pretraining, then K-means routing
python scripts/pretrain_manc.py --dataset PeMS04 --config configs/PeMS04.yaml

# Stage 2: train the dual-channel heterogeneous experts
python scripts/train.py --dataset PeMS04 --config configs/PeMS04.yaml

# Evaluate over all 12 prediction horizons
python scripts/evaluate.py --dataset PeMS04 --ckpt ckpt/PeMS04/best.pth
```

Stage 1 writes `ckpt/PeMS04/manc.pth` and `ckpt/PeMS04/routing.npz` (node-to-subset assignment, plus the subset indices). Stage 2 consumes that routing file; no re-clustering is performed, so results are reproducible from the saved assignment.

Run everything (four datasets, five seeds) with:

```bash
bash scripts/reproduce_all.sh
```

---

## Configuration

Shared settings across datasets: input length 12 → output length 12 (5-min steps), L = 4 Conv1D layers with C = 64 channels, GRU hidden size 64 with M = 1 layer, 4-layer MLP decoder, latent dimension D = 128, diffusion order S = 1, feed-forward dimension 256, batch size 16, at most 70 epochs with early stopping on validation MAE, Adam optimizer.

Per-dataset settings (as reported in Section 4.1):

| Dataset | lr | mask ratio η | Q (expanded dim) | memory bank | node subsets (high / low) | #samples P |
|---|---|---|---|---|---|---|
| PeMS03 | 1e-3 | 0.5 | 24 | 24 | 200 / 158 | 26,185 |
| PeMS04 | 1e-3 | 0.6 | 48 | 48 | 154 / 153 | 16,969 |
| PeMS07 | 1.5e-3 | 0.5 | 60 | 48 | 608 / 275 | 28,201 |
| PeMS08 | 1.5e-3 | 0.5 | 24 | 24 | 86 / 84 | 17,833 |

`Q` denotes the expanded feature dimension of the spatial projection (Eq. 12). All values live in `configs/*.yaml`; override any of them from the command line, e.g. `--mask_ratio 0.4` or `--diffusion_order 2`.

---

## Results

Mean MAE / RMSE / MAPE (%) over 12 horizons and five runs (seed 2024–2028). Lower is better.

| Dataset | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| PeMS03 | 15.12 | 26.09 | 15.45 |
| PeMS04 | 18.32 | 30.05 | 11.98 |
| PeMS07 | 19.15 | 32.58 | 7.92 |
| PeMS08 | 13.41 | 23.05 | 8.81 |

Minor deviations (≤ 0.1 in MAE) may occur due to GPU nondeterminism and library versions.

---

## Ablation and robustness

Ablation configurations from Table 5 (`--expert`, `--routing`, `--adjacency`):

| Configuration | Flags |
|---|---|
| NC-Expert (full) | `--expert both --routing cluster --adjacency memory+selfcorr` |
| Dual-channel, stable experts only | `--expert stable --routing cluster` |
| Dual-channel, dynamic experts only | `--expert dynamic --routing cluster` |
| Single-channel, stable expert | `--expert stable --routing none` |
| Single-channel, dynamic expert | `--expert dynamic --routing none` |
| Random node partition | `--routing random` |
| Swapped expert assignment | `--routing swap` |
| Cosine-similarity adjacency | `--adjacency cosine` |

```bash
python scripts/run_ablation.py --dataset PeMS08 --config configs/PeMS08.yaml
```

Missing-data robustness (Fig. 7–8):

```bash
# random missing: ratio 0.1 / 0.2 / 0.3
python scripts/run_robustness.py --dataset PeMS08 --missing_type random --missing_ratio 0.3

# consecutive missing: 3 / 6 / 12 steps at an overall 10% missing rate
python scripts/run_robustness.py --dataset PeMS08 --missing_type consecutive --missing_len 12

# ablation without masked reconstruction (NC-Expert w/o mask)
python scripts/train.py --dataset PeMS08 --config configs/PeMS08.yaml --use_masking false
```

Masking-ratio and diffusion-order sensitivity sweeps are available through `scripts/run_sensitivity.py` (`--mask_ratio in {0.3,0.4,0.5,0.6,0.7}`, `--diffusion_order in {1,2,3}`).

---

## Citation

```bibtex
@article{ncexpert,
  title   = {NC-Expert: A Node-Clustering-Guided Dual-Channel Heterogeneous Expert Network for Traffic Flow Prediction},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO}
}
```

Please also cite the datasets (Caltrans PeMS) and the baselines you compare against.

---

## License

Released under the [MIT License](LICENSE). The PeMS data are subject to Caltrans' terms of use and are **not** redistributed in this repository.

## Contact

For questions about the code or the paper, please open an issue or contact the authors (details will be added after peer review).

