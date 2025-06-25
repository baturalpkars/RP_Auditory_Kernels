# Research Project: Auditory Kernels for Representing Degraded Speech

This repository supports the Bachelor Research Project (https://github.com/TU-Delft-CSE/Research-Project?tab=readme-ov-file) titled  
**"Auditory Kernels for Representing Degraded Speech"** at TU Delft (2025).

It investigates how biologically-inspired auditory kernels—learned from clean speech—perform under realistic degradations, aiming to evaluate kernel **robustness**, **selectivity**, and **denoising potential**.

## Project Overview

- **Goal:** Use sparse coding (Matching Pursuit) with auditory kernels to reconstruct degraded speech and evaluate reconstruction quality.
- **Approach:**  
  - Train auditory kernels on clean TIMIT speech.
  - Degrade clean utterances with 4 types of noise:  
    `babble`, `train_coming`, `white_noise`, `airportAnnouncement`  
  - Test at SNR levels: **-5, 0, 5, 10 dB**
- **Evaluation metrics:**  
  - **PESQ** – perceptual quality  
  - **STOI** – speech intelligibility  
  - **ViSQOL** – perceptual similarity  
  - **SRR** – structural fidelity (signal-to-residual ratio)

## Key Components

> All Python scripts are designed to run independently.  
> Each file is well-commented and self-explanatory.

| File / Folder                          | Description                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------|
| `degrade_and_save.py`                 | Adds realistic noise to clean speech at multiple SNRs                      |
| `reconstruct_degraded.py`            | Reconstructs noisy signals using auditory kernels                          |
| `reconstruct_clean.py`               | Reconstructs original clean signals using same kernels                     |
| `reconstruct_noises.py`              | Runs kernel encoding on noise-only samples                                 |
| `kernel_analyzer.py`                 | Generates kernel usage plots & SRR metrics                                 |
| `metrics.py`                         | Implements STOI, PESQ, ViSQOL, SNR metrics                                 |
| `speech_analyzer.py`                 | Core class to calculate and export metric scores                           |
| `run_pesq.py / run_stoi.py / ...`    | Run evaluation pipeline per metric and export `.csv` results               |
| `plot_*.py`                          | All result plotting scripts (PESQ/STOI subplots, SRR curves, diff histograms) |
| `results/`                           | Stores metric outputs and visualizations                                   |

## Setup

### 1. Requirements

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

Make sure to also install Docker if you want to run ViSQOL evaluations (see below).

### 2. Kernel Dictionary

The auditory kernel dictionary (`kernels_15040.jld2`) used for Matching Pursuit is from the open-source repository:  
https://github.com/D1mme/rp_auditory_kernels

Download the file and place it under the `ExampleEncodingDecoding/` directory as `kernels_15040.jld2`.

### 3. ViSQOL Setup (Optional)

To compute **MOS-LQO** using ViSQOL, make sure Docker is installed and running.  
Pull the ViSQOL Docker image:

```bash
docker pull mubtasimahasan/visqol:v3
```

No local installation of ViSQOL is needed. The `ViSQOLMetric` class handles everything via Docker.

---

## Full Pipeline

### 1. Generate Degraded Speech

Run `degrade_and_save.py` to add real-world noise at multiple SNR levels.  
Each clean utterance is degraded with:

- Noise types: `babble`, `white_noise`, `airportAnnouncement`, `train_coming`
- SNR levels: `-5`, `0`, `5`, `10` dB

All degraded WAVs will be saved under `./degraded_speeches/{speaker_id}/`.

---

### 2. Reconstruct Using Auditory Kernels

- `reconstruct_degraded.py`: Sparse reconstructions of noisy signals
- `reconstruct_clean.py`: Reconstruct clean speech (baseline)
- `reconstruct_noises.py`: Analyze noise-only encoding

Each reconstruction saves:
- `reconstructed.wav`
- `encoded_waveform.pkl` (kernel encoding)
- `norm_list.npy` (residual tracking)
- diagnostic plots (waveforms, kernel histograms, SRR)

---

### 3. Metric Evaluation

For intelligibility, quality, and fidelity:

- `run_pesq.py`: PESQ (1–4.5 range)
- `run_stoi.py`: STOI (0–1 range)
- `run_visqol.py`: ViSQOL MOS-LQO
- Metrics compare:
  - Clean ↔ Reconstructed Clean
  - Degraded ↔ Reconstructed Degraded
  - Reconstructed Clean ↔ Reconstructed Degraded

Outputs saved as CSV files in `results/`.

---

### 4. Plotting

All plotting scripts are modular. You can customize:

- `SNR_LEVEL` or `TARGET_SNR`
- `noise_types` to include
- `comparison_columns`

Example scripts:
- `plot_metric_subplots.py`: STOI/PESQ by SNR
- `plot_srr_curves.py`: SRR vs. kernel rate curves
- `plot_kernel_diff.py`: Normalized kernel activation differences

All figures saved under `results/plots`.

---

## Reproducibility

All code is:

- Deterministic (no random seeds unless noted)
- Compatible with Python 3.8+
- Modular and self-contained — no need to run everything at once
- Fully commented and logically grouped by task

---

## Responsible Research

- All data is from the public **MS-SNSD** corpus (consented, anonymized).
- Noise is synthetic; no biometric/speaker recognition used.
- Code is open-source and available at [github.com/baturalpkars/RP_Auditory_Kernels](https://github.com/baturalpkars/RP_Auditory_Kernels)

---

## Contact

For questions or ideas:

**Baturalp Karslıoğlu**  
TU Delft – Bachelor Research Project  
email: b.karslioglu@student.tudelft.nl