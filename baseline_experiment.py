BIOLOGICAL_CONTEXT = """This dataset comprises single-cell data of human prostate adenocarcinoma samples, including both tumor and benign tissues."""
# for prostate we only got on group which is Tumor
import json
from pathlib import Path
import uuid
from call_gpt import OpenAiClient, GeneSelection
from executenote import extract_python_code, execute_code_in_notebook, read_last_n_results
from utils_agent import fetch_string_neighbors_clean, fetch_gene_summary
import json
import os
def load_json_context(json_path: str):
    """
    Load structured context from a JSON file.
    Returns a Python dict / list depending on JSON content.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_text_content(txt_path: str):
    """
    Load raw content from a text file.
    Returns a string.
    """
    txt_path = Path(txt_path)
    with txt_path.open("r", encoding="utf-8") as f:
        return f.read()
    
analysis_tool = """
functions that would return an updated version of adata would have  -> return adata at the end. You should use adata = function() for that
You should import these functions from utils_agent. You should also import scanpy, pandas, numpy, and matplotlib.pyplot.
for function returning None you should use them directly without trying to return adata.

def filter_cells(adata, min_cells=3, min_genes=200, max_genes=50000) -> return None
def normalize_log_transform(adata, target_sum = 1e4) -> return None
def filter_lowqc_cells(adata, pct_counts_mt_upbound = 10, n_genes_by_counts = None, pct_counts_mt_lowbound = None) -> return adata
def save_high_var_gene(adata, n_top_genes = 4000) -> return adata
def pca_and_plot_umap(
    adata, 
    n_pcs=40, 
    n_neighbors=10, 
    resolution=0.7,
) -> return None
def annotate_with_celltypist_safe(adata)
def fetch_string_neighbors_clean(gene_symbol)
def fetch_gene_summary(gene_symbol)
"""

prompt_analysis = """Now you are given an adata. You should load it from {adata_path}, conduct quality control, filtering and hvg as required. Then you should do PCA and UMAP, conduct 
annnotation with the given Celltypist function, do a crosstab to examine numbers of samples in each group by different cluster. output your crosstab result to crosstab.txt.
do a DEG (note that DEG results are cluster-wise), and extract candidate genes across clusters
using sc.get.rank_genes_groups_df.

From these DEG results, select ~30 representative genes across all clusters
(by taking top genes per cluster and then de-duplicating by gene name).

For these ~30 genes, check adata.obs['celltypist_major'] for their annotation and also record
the specific value for their pval, score, and logFC.

Optionally use pathway or GO enrichment results to help select the final ~30 genes
from the DEG candidates in a principled, non-cluster-specific way.

use the given function to fetch gene summary and their protein neighbors, and output all these information into a JSON file. You should have ~30 potential genes into your JSON file eventually.
with regard to the following format:
named final.json with each gene as keys and all the metadata with corresponding values as key-value pair in dictionary corresponding to the gene.
You are provided with the following tool:
{tools}
Note:
- sc.tl.rank_genes_groups stores DEG results per cluster; it is NOT a flat gene list.
- You should use sc.get.rank_genes_groups_df(adata, group=None) to extract DEG results.
- Do not assume adata.uns['rank_genes_groups']['names'] is a one-dimensional list.
"""

prompt_selection = """You are given a crosstab result, a biological context of the data sample, and a JSON file contains the analysis result.
You are supposed to select 10 target genes from the given information and for each of them you should give specific reasoning referencing to the 
information and figures provided. 
biological_context:
{context}
crosstab_result:
{crosstab}
json_file:
{json}
"""

curr_analysis_prompt = prompt_analysis.format(adata_path = "tumor_adata.h5ad", tools = analysis_tool)
print(curr_analysis_prompt)
from call_gpt import OpenAiClient

client = OpenAiClient()

TEST_FILE_NAME = "baseline_analysis.ipynb"
current_code = extract_python_code(client.call_openai_gpt(prompt=curr_analysis_prompt))
result = execute_code_in_notebook(TEST_FILE_NAME, current_code)
# # # when files are ready
curr_prompt_selection = prompt_selection.format(context = BIOLOGICAL_CONTEXT, crosstab = load_text_content("crosstab.txt"), json = load_json_context("final.json"))
client.call_openai_gpt(prompt=curr_prompt_selection)
