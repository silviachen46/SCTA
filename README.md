# SCTA: An Agentic Framework for Stable and Interpretable Target Gene Discovery from Single-Cell RNA Sequencing

## Overview
This repository contains the source code for the paper:

**SCTA: An Agentic Framework for Stable and Interpretable Target Gene Discovery from Single-Cell RNA Sequencing**

<img width="727" height="294" alt="image" src="https://github.com/user-attachments/assets/25e43577-4936-4a76-9a09-a5ca66baf9ab" />

**Abstract**
Diseases with high heterogeneity often exhibit cellular-level variation that cannot be adequately captured by bulk transcriptomic approaches, making single-cell RNA sequencing (scRNA-seq) a powerful modality for dissecting disease endotypes. However, identifying biologically meaningful target genes from scRNA-seq data remains challenging due to the complex, multi-stage, and decision-dependent nature of single-cell analysis workflows, which are further complicated by diverse dataset structures and metadata. We present **SCTA** (Single-Cell Target Agent), a modular, fully autonomous multi-agent framework for end-to-end target gene discovery in single-cell data. SCTA decomposes the analysis into specialized agents with distinct analytical roles and integrates biological knowledge as a structural constraint to guide downstream reasoning. Across multiple independent runs, the framework demonstrates stable convergence of target gene selection, as quantified by selection frequency and pairwise overlap metrics. Representative case studies further show that SCTA can recover biologically coherent and disease-specific mechanisms. Together, this work highlights the agent-based orchestration to enable interpretable, robust, and scalable target gene discovery in single-cell-driven precision medicine.

**Keywords**
single-cell RNA sequencing,
target gene discovery,
agentic framework,
computational biology


The code implements the full pipeline described in the manuscript, including preprocessing, agent coordination, target gene discovery, and evaluation.  
All results reported in the paper can be reproduced using the instructions below.

---

## Repository Structure
```
├── baseline/ # Baseline method scripts and runners
├── biomni_script/ # Biomni baseline running scripts
├── covid_sample_run/ # Example run artifacts
│ └── sample_result_files/ # Sample output files produced by the pipeline
├── agent_v5.py # Core agent orchestration / pipeline logic
├── call_gpt.py # LLM client wrapper (OpenAI/Azure selection)
├── executenote.py # Notebook execution / orchestration utilities
├── main.py # Main entry point
├── utils_agent.py # Shared utilities used by agents/pipeline
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

```

---

## Environment Setup

### System Requirements
- Operating system: macOS (tested on macOS Tahoe 26.2)
- Python version: [e.g., Python 3.13.1]
- Hardware: [CPU-only]

### Install Dependencies

```bash
conda create -n scta python=3.13.1
conda activate scta
pip install -r requirements.txt
```
Register kernel space for automatically running the notebook:
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name python3 --display-name "Python 3 (scta)"
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
use this curl command to download the gmt file required for enrichment:

Human

```bash
curl -L \
"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human" \
-o KEGG_2021_Human.gmt
```

Mouse
```bash
curl -L \
"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2019_Mouse" \
-o KEGG_mouse_2019.gmt
```



Place all of them in your root folder for easier management.

### Step 1: Replacing necessary configurations
## agent_v5.py:

1. replace BIOLOGICAL_CONTEXT with the following context corresponding to your adata
```bash
# Prostate adenocarcinoma (GSE193337)
BIOLOGICAL_CONTEXT = """This dataset comprises single-cell data of human prostate adenocarcinoma samples, including both tumor and benign tissues."""

# Chronic pancreatitis (GSE165045)
BIOLOGICAL_CONTEXT = """This dataset profiles pancreatic immune cells from healthy donors and patients with chronic pancreatitis of different etiologies (hereditary and idiopathic) using single-cell transcriptomic and surface protein measurements."""

# COVID-19 (GSE149689)
BIOLOGICAL_CONTEXT = """The dataset contains single-cell RNA sequencing profiles of peripheral blood mononuclear cells collected from healthy individuals, patients with mild COVID-19, patients with severe COVID-19, and patients with severe influenza""" 

```

2. Replace the following placeholders with your custom values:

- `TEST_FILE_NAME`  
  Name of the Python notebook used in the pipeline (must end with `.ipynb`).

- `GLOBAL_RESULT`  
  Path to the global result file (must be a `.txt` file).  
  This file records intermediate results and can be used to inspect the execution process.

- `ADATA_SOURCE_PATH`  
  Absolute path to the selected AnnData (`.h5ad`) file used as input.

In addition, specify the client type by setting:

- `CLIENT_TYPE`  
  Choose `"GPT"` to directly use the OpenAI API, or `"Azure"` to use the Azure OpenAI service.


## call_gpt.py:

replace your api_key in OpenAI Client. If you select "Azure" in your above step, you should also specify api_key, api_version, and azure_endpoint in AzureOpenAI Client.

## utils_agent.py:
replace
```bash
Species = "Human" # or "Mouse"
```
and gmt file absolute path map for corresponding species.

for example:
```bash
enrich_kmt_file_map = {
    "Human" : "/Users/silviachen/Documents/Software/new_sca_agent/SCAagent/KEGG_2021_Human.gmt",
    "Mouse" : "/Users/silviachen/Documents/Software/SCAagent/KEGG_mouse_2019.gmt"
}
```

### Step 2: Agent-based Analysis
Once everything is ready, simply run the script using `python main.py`. You should see logs start printing in terminal.

### Step 3: Checking the Results

After the pipeline finishes, the following files and folders are expected to be generated:

#### Interpreting Output Files
- `deg_tmp_results.txt`  
  Intermediate differential expression results.

- `graph_results.txt`  
  Graph-based analysis results (file name may vary depending on configuration).

- Jupyter notebook(s)  
  Notebooks documenting the agentic steps used to conduct the single-cell analysis.

#### Output Directories
- `potential_gene_set/`  
  Contains the initially selected candidate target genes for each disease group.

- `result_gene_set/`  
  Contains the final selected target genes for each disease group.

Each of the two directories includes `disease_group.json` files corresponding to different disease groups.

---

## License
MIT License

---

## Contact
**Corresponding Author**  
Name: Haohan Wang
Email: [haohanw@illinois.edu]

**Authors**  
Name: Shuyu Chen, Ye Zhang, Yang Li, Qiqi Xie
Email: [shuyu5@illinois.edu]
For questions regarding code or data, feel free to reach out via email.

