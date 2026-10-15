# Rain Prediction — South America (ERA5)

Kaggle competition: [previsao-climatica-de-precipitacao-sobre-a-america-do-sul](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul)

Given the ERA5 atmospheric state of month **M**, predict the mean precipitation field (mm/day) for month **M+1** over South America.

---

## Architecture

**ConvLSTM + Conditional DDPM (Diffusion Model)**

```
Input: 6 months × 9 atmospheric variables (301 × 261 grid)
         │
         ▼
  ConvLSTMCell  ──── processes each month sequentially
         │
         ▼
  Context h  (64 × 301 × 261)
         │
         ▼  concatenated with noisy target
  U-Net (3 levels, GroupNorm + SiLU)  ──── conditioned on h + timestep embedding
         │
         ▼
  Predicted noise  →  DDIM reverse  →  Precipitation anomaly
         │
         ▼
  + Monthly climatology  →  Final precipitation (mm/day)
```

### Components

| Component | Details |
|-----------|---------|
| `ConvLSTMCell` | Replaces FC gates with Conv2d; encodes spatial-temporal dynamics |
| `ConditionalUNet` | 3 encoder levels (32→64→128 ch), skip connections, bilinear upsampling |
| Normalization | GroupNorm (stable with small batch sizes) |
| Activation | SiLU |
| Noise schedule | Cosine (1000 timesteps) |
| Loss | `L_noise + 0.5 × L_x0` (hybrid: noise prediction + direct reconstruction) |
| Inference | DDIM (50 steps), mean of 40 samples |

---

## Training Strategy

### Input
- **9 ERA5 variables** per month (precipitation excluded from inputs):  
  `t2m`, `cloud_cover`, `geopotential_850`, `rel_hum_850`, `shum_850`, `surface_pressure`, `temperature_850`, `u_850`, `v_850`
- **seq_len = 6** months of context

### Target
- **Precipitation anomaly**: `y - clim(month)`
- `clim(month)` = 44-year (1979–2022) pixel-wise monthly mean
- Model learns only interannual variability; the seasonal cycle is added back at inference

### Why no precipitation in inputs?
The test set provides real atmospheric features for all 24 months (Jan 2023 – Dec 2024) but not precipitation. Including past precipitation would force autoregressive prediction, accumulating errors across months. Excluding it allows each of the 24 months to be predicted independently using 100% real inputs.

### Optimization
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | `2e-4` → `1e-6` (cosine annealing) |
| Epochs | 50 |
| Batch size | 4 |
| Gradient clipping | 1.0 |

---

## Scores

| Iteration | Change | Kaggle RMSE |
|-----------|--------|------------|
| v1 | Baseline DDPM, 10 epochs, linear schedule, 10 channels | 4.20 |
| v2 | Remove tp from inputs (no autoregression) | 3.57 |
| v3 | 3-level U-Net, GroupNorm, cosine schedule, hybrid loss, z-score norm | 3.00 |
| v4 | seq_len=6, epochs=50, ddim_steps=50, 40 inference samples | 2.66 |
| v5 | Monthly climatology anomaly target | TBD |

---

## Setup

```bash
pip install torch numpy pandas xarray netcdf4 h5netcdf tqdm
```

Place all data files inside `data/`:

```
WorCap_RAIN-PREDICTION/
├── solution.py
├── README.md
└── data/
    ├── treino_tp.nc
    ├── treino_tp_alvo.nc
    ├── treino_t2.nc
    ├── treino_cloud_cover.nc
    ├── treino_geopotential_850.nc
    ├── treino_rel_hum_850.nc
    ├── treino_shum_850.nc
    ├── treino_surface_pressure.nc
    ├── treino_temperature_850.nc
    ├── treino_u_850.nc
    ├── treino_v_850.nc
    ├── teste_features.nc
    └── sample_submission.csv
```

Download from Kaggle CLI:
```bash
kaggle competitions download -c previsao-climatica-de-precipitacao-sobre-a-america-do-sul
```

Credentials: `C:\Users\<user>\.kaggle\kaggle.json` ([generate here](https://www.kaggle.com/settings)).

---

## Usage

```python
# In solution.py — set data_dir in CONFIG:
# Local
'data_dir': r'C:\Users\vinic\OneDrive\Desktop\WorCap_RAIN-PREDICTION\data'
# Kaggle
'data_dir': '/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul'
```

```bash
python solution.py
```

This will:
1. Train the model for `epochs` iterations
2. Run local RMSE validation on the last 12 months of training data
3. Generate `submission.csv` ready for upload

---

## Key Files

| File | Description |
|------|-------------|
| `solution.py` | Full pipeline: dataset, model, training, inference |
| `modelo_final.pth` | Saved model weights (generated after training) |
| `norm_stats.npy` | Normalization stats + climatology (generated after training) |
| `submission.csv` | Final predictions (generated after inference) |
