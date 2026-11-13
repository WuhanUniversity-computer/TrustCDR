# TrustCDR: Reliability-Guided CDR Co-Design under Local Antigen Uncertainty

TrustCDR extends [FlowDesign](https://github.com/BioDesign-center/FlowDesign) / DiffAb with a **reliability-aware (RA)** framework that guides CDR co-design when the antigen local conformation is uncertain. The core idea: not all antigen epitope residues are equally trustworthy — TrustCDR estimates per-residue reliability from an ensemble of perturbed antigen conformations, then injects this signal into the antibody design generator via a series of lightweight, pluggable modules.

---

## Method Overview

TrustCDR sits on top of the Rectified Flow antibody design backbone. Three RA modules are added:

| Module | Purpose | Key files |
|--------|---------|-----------|
| **Reliability Field** | Quantify per-antigen-residue reliability via an ensemble of locally perturbed conformations, scored across coordinate / topology / orientation / surface channels | `diffab/ra_flowdesign/reliability.py` |
| **Context Encoder** | Encode the reliability field + optional antigen sequence context into residue-wise embeddings | `diffab/ra_flowdesign/reliability_context_encoder.py` · `antigen_sequence_context.py` |
| **Relation Gate & Pair Modulation** | Inject reliability into the pair-update stream of the flow generator — a gating mechanism attenuates noisy antigen contact signals, while a residual correction branch adjusts pair features | `diffab/ra_flowdesign/relation_gate.py` · `pair_modulation*.py` |

The model is trained with staged checkpoints (200k → 250k → 300k steps) and evaluated under both **clean** and **Backrub-perturbed** antigens to measure robustness.

---



---

## Environment

```bash
conda env create -f env.yaml -n flow
conda activate flow
```

Dependencies: Python 3.8 · PyTorch 1.12.1 · CUDA 11.3 · BioPython · abnumber · mmseqs2 · pdbfixer. PyRosetta + Ray are optional (used for structure relaxation / binding energy).

---

## Data

### Training data (SAbDab)
1. Download `all_structures.zip` from [SAbDab archive](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/).
2. Extract into `data/`.
3. Download the template bundle ([Google Drive](https://drive.google.com/file/d/1LFnvSkvUWl8fjAxOfGus1Qf9QnCirq-M/view?usp=drive_link)) and extract `template.zip` into the project root (Chothia-numbered PKLs + IMGT renumbered PDBs used for template construction).

### Perturbation robustness benchmark
Backrub-perturbed antigen structures are generated with `scripts/generate_backrub_perturbations.py` and evaluated through `evaluation/run_shared_perturb_eval.py`.

---

## Usage

### 1. Training

```bash
python train.py -c configs/train/<your_config>.yaml
```

RA-related knobs live in `config.py`:
- `use_ra_flowdesign`: master switch
- `ra_num_ensemble`, `ra_perturb_sigma`, `ra_reliability_channels`: reliability field hyperparameters
- `ra_gate_mode`, `ra_gate_insert_attention`, `ra_gate_insert_pair_update`: relation gate placement
- `ra_context_encoder_mode`, `ra_context_layers`: context encoder depth
- `ra_use_denoising_head`: auxiliary denoising objective

Param-group learning rates (backbone vs. gate vs. FiLM vs. pair-modulation) are defined in `train.py`'s `get_param_group_optimizer`.

### 2. Inference

#### Test set
```bash
python design_testset.py --template_dict <path_to_dict.pkl> \
    -c configs/test/<config>.yaml -b 32 1
```
Batch scripts: `bash test_all.sh`, `bash test_hiv.sh` (HIV antibody sampling).

#### Single PDB
```bash
python design_pdb.py <path_to_pdb> \
    --heavy <H> --light <L> \
    --template <template.pkl> \
    --config configs/test/<config>.yaml
```

### 3. Perturbation Robustness Evaluation

```bash
# Run full perturbation eval with a given checkpoint
python evaluation/run_shared_perturb_eval.py \
    -c configs/test/shared_perturb_eval.yml \
    -o eval_results/my_exp

# Multi-seed robustness analysis
bash scripts/multiseed_inference_robustness_all_methods/compute_raflowdesign_all_samples_dockq.py
```

### 4. Reliability Cache (recommended for speed)

```bash
# Precompute reliability fields for your test set
python scripts/precompute_antigen_reliability.py \
    -o cache/antigen_reliability/

# Enable cache in config:
# use_antigen_reliability_cache: True
# antigen_reliability_cache_dir: cache/antigen_reliability/
```

---



## Released Checkpoints

| Path | Description |
|------|-------------|
| `trained_models/cdrh3.pt` | FlowDesign CDRH3 generator |
| `trained_models/hiv.pt` | HIV antibody sampling model |

TrustCDR checkpoint names follow the pattern `ra_pair_gate_fix_scratch_0_to_40w` / `softC_film_v2_consistency_15w_to_25w` in the training logs.

---

## License

See `LICENSE`.
