# Alignment Methods Assignment - Complete Implementation

Comprehensive implementation of DPO, PPO, and GRPO alignment methods with reward hacking analysis and evaluation pipeline.

##  Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Ablation Experiments](#ablation-experiments)
- [Analysis](#analysis)

---

##  Overview

This project implements three state-of-the-art alignment methods:

1. **DPO (Direct Preference Optimization)** - Direct optimization on preference pairs without reward model
2. **PPO (Proximal Policy Optimization)** - RL-based alignment with sparse/dense rewards
3. **GRPO (Group Relative Policy Optimization)** - Group-based optimization without value function

All methods are evaluated on:
- **Alignment quality** (reward scores, preference accuracy)
- **Alignment tax** (perplexity, catastrophic forgetting)
- **Failure modes** (reward hacking, verbosity bias)

---

##  Project Structure

```
alignment-project/
├── config/
│   └── default_config.py          # Comprehensive configuration with argparse
├── data/
│   ├── raw/                        # Raw ORCA DPO Pairs dataset
│   └── processed/                  # Processed train/val/test splits
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       ├── instr_following_subset.jsonl
│       └── statistics.json
├── models/
│   ├── baseline/                   # SmolLM2-135M-Instruct (SFT)
│   ├── reward_model/               # Trained reward model
│   ├── dpo/                        # DPO checkpoints
│   ├── ppo/                        # PPO checkpoints
│   └── grpo/                       # GRPO checkpoints
├── eval/
│   ├── testset_50.jsonl           # 50-prompt test set
│   ├── *_outputs.jsonl            # Generated outputs
│   ├── *_metrics.csv              # Computed metrics
│   └── hack_tests.csv             # Reward hacking results
├── runs/                           # Training run logs
├── notebooks/                      # Analysis notebooks
├── scripts/
│   ├── prepare_data.py            # Data preprocessing
│   ├── train_reward_model.py     # Reward model training
│   ├── train_dpo.py               # DPO training
│   ├── train_ppo.py               # PPO training (sparse/dense)
│   ├── train_grpo.py              # GRPO training
│   ├── eval_generate.py           # Generate model outputs
│   ├── eval_metrics.py            # Compute evaluation metrics
│   └── perturbations.py           # Reward hacking tests
└── README.md                       # This file
```

---

##  Setup

### Easy Installation (Recommended!)

**Option 1: Use the automated bash script**
```bash
# Run the setup script (creates virtual environment and installs everything)
bash setup.sh

# Or for quick install (skip optional packages)
bash setup.sh --quick

# Or for CPU-only
bash setup.sh --cpu-only
```

**Option 2: Use the Python installer**
```bash
# Run the Python installer
python install.py

# Or for quick install
python install.py --quick

# Or for CPU-only
python install.py --cpu-only
```

**Option 3: Manual installation with requirements.txt**
```bash
# Create virtual environment (recommended)
python3 -m venv alignment_env
source alignment_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# For CPU-only (skip bitsandbytes)
grep -v "bitsandbytes" requirements.txt | pip install -r /dev/stdin
```

**Option 4: Manual installation (individual packages)**
```bash
# Core packages
pip install torch transformers datasets peft trl accelerate bitsandbytes
pip install numpy pandas matplotlib seaborn tqdm scikit-learn scipy

# Optional but recommended
pip install tensorboard jupyter notebook
```

### Verify Installation

```bash
# Using setup script
bash setup.sh --verify-only

# Using Python installer
python install.py --verify

# Or manually
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### What Gets Installed

The installation scripts install:
-  **PyTorch** (with CUDA support if available)
-  **Transformers** & HuggingFace libraries
-  **PEFT** (LoRA fine-tuning)
-  **TRL** (RLHF training)
-  **bitsandbytes** (8-bit quantization)
-  **Scientific libraries** (NumPy, Pandas, SciPy, scikit-learn)
-  **Visualization** (Matplotlib, Seaborn, TensorBoard)
-  **Utilities** (tqdm, requests, pyarrow)

---

##  Quick Start

### Option 1: Run Everything with main.py (Recommended!)

The easiest way to run all experiments is using the orchestration script:

```bash
# Run complete pipeline with defaults (3 seeds, 3 epochs)
python main.py --full_pipeline

# Quick test mode (1 seed, 1 epoch - for testing)
python main.py --quick_test

# Run with custom configuration
python main.py --full_pipeline --seeds 42 123 456 --epochs 5 --batch_size 16

# Include ablation studies
python main.py --full_pipeline --run_ablations
```

**What it does:**
1.  Prepares data
2.  Trains reward model
3.  Trains DPO (all seeds)
4.  Trains PPO sparse (all seeds)
5.  Trains PPO dense (1 seed)
6.  Trains GRPO (all seeds)
7.  Runs evaluation pipeline
8.  Tests reward hacking
9.  Saves comprehensive results

### Option 2: Run Specific Stages

```bash
# Run only data preparation
python main.py --stage data

# Run only reward model training
python main.py --stage reward_model

# Run only alignment methods
python main.py --stage alignment

# Run only evaluation
python main.py --stage evaluation
```

### Option 3: Manual Step-by-Step

### 1. Data Preparation

```bash
python scripts/prepare_data.py \
  --seed 42
```

**Outputs:**
- `data/processed/train.jsonl` (train split)
- `data/processed/val.jsonl` (validation split)
- `data/processed/test.jsonl` (test split)
- `data/processed/instr_following_subset.jsonl` (for perplexity)
- Statistics and length distributions

### 2. Train Reward Model

```bash
python scripts/train_reward_model.py \
  --batch_size 16 \
  --lr 5e-5 \
  --epochs 3 \
  --lora_r 8 \
  --seed 42 \
  --save_dir ./models/reward_model
```

**Outputs:**
- Trained reward model
- Validation metrics (AUC, accuracy)
- Reward statistics (chosen vs rejected margin)
- Visualization plots

### 3. Train Alignment Methods

#### DPO

```bash
python scripts/train_dpo.py \
  --method DPO \
  --batch_size 8 \
  --lr 1e-4 \
  --epochs 3 \
  --lora_r 8 \
  --beta 0.1 \
  --seed 42 \
  --save_dir ./checkpoints/dpo
```

#### PPO (Sparse Reward)

```bash
python scripts/train_ppo.py \
  --method PPO \
  --reward_model_path ./models/reward_model/final_model \
  --reward_mode sparse \
  --batch_size 8 \
  --lr 1e-5 \
  --kl_coef 0.05 \
  --epochs 3 \
  --seed 42 \
  --save_dir ./checkpoints/ppo_sparse
```

#### PPO (Dense Reward)

```bash
python scripts/train_ppo.py \
  --method PPO \
  --reward_model_path ./models/reward_model/final_model \
  --reward_mode dense \
  --batch_size 8 \
  --lr 1e-5 \
  --kl_coef 0.05 \
  --epochs 3 \
  --seed 42 \
  --save_dir ./checkpoints/ppo_dense
```

#### GRPO

```bash
python scripts/train_grpo.py \
  --method GRPO \
  --reward_model_path ./models/reward_model/final_model \
  --group_size 8 \
  --advantage_normalization rank \
  --batch_size 4 \
  --lr 1e-5 \
  --epochs 3 \
  --seed 42 \
  --save_dir ./checkpoints/grpo
```

---

## Evaluation Pipeline

### 1. Generate Outputs

Generate responses for test prompts:

```bash
python scripts/eval_generate.py \
  --model_path ./checkpoints/dpo/final_model \
  --test_file ./eval/testset_50.jsonl \
  --output_file ./eval/dpo_outputs.jsonl \
  --temperature 0.7 \
  --seed 42
```

### 2. Compute Metrics

Compute comprehensive metrics on generated outputs:

```bash
python scripts/eval_metrics.py \
  --generated_file ./eval/dpo_outputs.jsonl \
  --reward_model_path ./models/reward_model/final_model \
  --reference_model_path HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy_model_path ./checkpoints/dpo/final_model \
  --output_file ./eval/dpo_metrics.csv \
  --seed 42
```

**Metrics Computed:**
- Reward scores
- KL divergence (vs reference)
- Perplexity
- Length statistics
- Compliance rates (for constrained prompts)
- Quality proxies

### 3. Test Reward Hacking

Test for reward hacking vulnerabilities:

```bash
python scripts/perturbations.py \
  --model_path ./checkpoints/ppo_sparse/final_model \
  --reward_model_path ./models/reward_model/final_model \
  --test_file ./eval/testset_50.jsonl \
  --output_file ./eval/ppo_hack_tests.csv \
  --seed 42
```

**Tests:**
- Surface-form perturbations (filler phrases, reordering)
- Alignment keyword injection
- Adversarial prompts
- Length vs reward correlation

---

## Ablation Experiments

### DPO Ablations

```bash
# Learning rate sweep
for lr in 1e-4 5e-5 1e-5; do
  python scripts/train_dpo.py \
    --lr $lr \
    --seed 42 \
    --save_dir ./checkpoints/dpo_lr_${lr}
done

# LoRA rank sweep
for rank in 4 8 16; do
  python scripts/train_dpo.py \
    --lora_r $rank \
    --seed 42 \
    --save_dir ./checkpoints/dpo_rank_${rank}
done

# Beta sweep
for beta in 0.05 0.1 0.2; do
  python scripts/train_dpo.py \
    --beta $beta \
    --seed 42 \
    --save_dir ./checkpoints/dpo_beta_${beta}
done
```

### PPO Ablations

```bash
# KL coefficient sweep
for kl in 0.01 0.05 0.1; do
  python scripts/train_ppo.py \
    --reward_model_path ./models/reward_model/final_model \
    --kl_coef $kl \
    --seed 42 \
    --save_dir ./checkpoints/ppo_kl_${kl}
done

# No-KL ablation (shows drift)
python scripts/train_ppo.py \
  --reward_model_path ./models/reward_model/final_model \
  --kl_coef 0.0 \
  --seed 42 \
  --save_dir ./checkpoints/ppo_no_kl

# Large-KL ablation (shows constraint)
python scripts/train_ppo.py \
  --reward_model_path ./models/reward_model/final_model \
  --kl_coef 0.5 \
  --seed 42 \
  --save_dir ./checkpoints/ppo_large_kl

# Temperature sweep
for temp in 0.2 0.7 1.0; do
  python scripts/train_ppo.py \
    --reward_model_path ./models/reward_model/final_model \
    --temperature $temp \
    --seed 42 \
    --save_dir ./checkpoints/ppo_temp_${temp}
done
```

### GRPO Ablations

```bash
# Group size sweep
for gs in 4 8 16; do
  python scripts/train_grpo.py \
    --reward_model_path ./models/reward_model/final_model \
    --group_size $gs \
    --seed 42 \
    --save_dir ./checkpoints/grpo_g${gs}
done

# Normalization method comparison
for norm in rank zscore minmax; do
  python scripts/train_grpo.py \
    --reward_model_path ./models/reward_model/final_model \
    --advantage_normalization $norm \
    --seed 42 \
    --save_dir ./checkpoints/grpo_${norm}
done
```

### Multi-Seed Experiments

```bash
# Run each method with 3 seeds
for seed in 42 123 456; do
  # DPO
  python scripts/train_dpo.py --seed $seed --save_dir ./checkpoints/dpo_seed_${seed}
  
  # PPO
  python scripts/train_ppo.py \
    --reward_model_path ./models/reward_model/final_model \
    --seed $seed \
    --save_dir ./checkpoints/ppo_seed_${seed}
  
  # GRPO
  python scripts/train_grpo.py \
    --reward_model_path ./models/reward_model/final_model \
    --seed $seed \
    --save_dir ./checkpoints/grpo_seed_${seed}
done
```

---

## Analysis

### Key Metrics to Compare

| Metric | Description | Goal |
|--------|-------------|------|
| **Reward Score** | Mean reward from reward model | Higher is better |
| **Perplexity** | On instruction-following data | Lower is better (less forgetting) |
| **KL Divergence** | Policy drift from reference | Lower is better (less drift) |
| **Compliance Rate** | Following length constraints | Higher is better |
| **Hacking Sensitivity** | Reward delta under perturbations | Lower is better |

<!-- ### Expected Findings

**DPO:**
- ✅ Stable training
- ✅ Low computational cost
- ✅ Good preference alignment
- ⚠️ Limited flexibility

**PPO (Sparse):**
- ✅ Flexible reward shaping
- ⚠️ Requires value function
- ⚠️ Can overoptimize reward
- ⚠️ Sensitive to KL coefficient

**PPO (Dense):**
- ✅ Better credit assignment
- ⚠️ More computationally expensive
- ⚠️ Can still overoptimize

**GRPO:**
- ✅ No value function needed
- ✅ More stable than PPO
- ✅ Group-relative advantages
- ⚠️ Requires multiple samples

### Failure Modes

**Reward Hacking:**
- Models exploit reward model weaknesses
- High reward but low actual quality
- Detected via perturbation sensitivity

**Verbosity Bias:**
- Models produce unnecessarily long responses
- Length correlates with reward
- Detected via compliance rate and length statistics

**Catastrophic Forgetting:**
- Loss of original capabilities
- High perplexity on SFT data
- Large KL divergence from reference

--- -->

## Acknowledgments

- **SmolLM2** - HuggingFace base model
- **ORCA DPO Pairs** - Preference dataset
- **TRL** - Training library for RLHF
- **PEFT** - Parameter-efficient fine-tuning
