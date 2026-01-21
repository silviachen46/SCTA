# SCTA: An Agentic Framework for Stable and Interpretable Target Gene Discovery from scRNA-seq

## Overview
This repository contains the source code for the paper:

**SCTA: An Agentic Framework for Stable and Interpretable Target Gene Discovery from Single-Cell RNA Sequencing**

The code implements the full pipeline described in the manuscript, including preprocessing, agent coordination, target gene prioritization, and evaluation.  
All results reported in the paper can be reproduced using the instructions below.

---

## Repository Structure
```
├── data/                  # Input data (or scripts to download data)
├── preprocessing/         # Single-cell preprocessing code
├── agents/                # Agent implementations
├── evaluation/            # Evaluation and ablation analysis
├── experiments/           # Scripts to reproduce paper experiments
├── configs/               # Configuration files
├── scripts/               # Entry-point scripts
├── figures/               # Code to generate figures
├── environment.yml        # Conda environment specification
├── requirements.txt       # Optional pip requirements
└── README.md
```

---

## Environment Setup

### System Requirements
- Operating system: macOS (tested on macOS Tahoe 26.2)
- Python version: [e.g., Python 3.13.1]
- Hardware: [CPU-only]

### Install Dependencies

using pip:
```bash
pip install -r requirements.txt
```

---

## Data Availability

### Datasets Used in the Paper
- Dataset 1: [Prostate adenocarcinoma(GSE193337)]  
  Source: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165045]

- Dataset 2: [Chronic pancreatitis\\(GSE165045)]  
  Source: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149689]
  
- Dataset 3: [COVID-19(GSE149689)]  
  Source: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193337]
  
You can download the preprocessed(label with sample number and disease groups) here: [https://drive.google.com/drive/folders/1XFvLRB8b_ESJhlwW0TmHRgH_JmNZzf7Q?usp=sharing]
You should place them in the root folder before starting the agents
---

## Running the Pipeline

### Step 1: Preprocessing
```bash
python scripts/run_preprocessing.py --config configs/preprocess.yaml
```

### Step 2: Agent-based Analysis
```bash
python scripts/run_agents.py --config configs/agents.yaml
```

### Step 3: Target Gene Selection
```bash
python scripts/run_target_selection.py --config configs/targets.yaml
```

---

## Reproducing Paper Results


---

## Configuration Details
All experiments are controlled via YAML configuration files located in `configs/`.  
Parameters corresponding to the paper are documented inline.

---

## Reproducibility Notes
- Random seeds are fixed where applicable.
- Software versions are pinned.
- Hyperparameters match the manuscript unless otherwise stated.

---

## License
[Specify license]

---

## Contact
**Corresponding Author**  
Name: Haohan Wang
Email: [haohanw@illinois.edu]

**First Author**  
Name: Shuyu Chen
Email: [shuyu5@illinois.edu]
