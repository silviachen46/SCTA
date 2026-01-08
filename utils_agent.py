from typing import Dict, List, Union
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
import os, re
import warnings
from anndata import ImplicitModificationWarning
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from itertools import product
import celltypist
import scanpy as sc
import json
import requests
# load and mark

import scanpy as sc
import pandas as pd



# path specification for Human/Mouse
# TO MODIFY BEFORE USE
deg_tmp_file_name = "deg_tmp_result.txt"
Species = "Mouse" # or "Mouse"
enrich_kmt_file_map = {
    "Human" : "/Users/silviachen/Documents/Software/new_sca_agent/SCAagent/KEGG_2021_Human.gmt",
    "Mouse" : "/Users/silviachen/Documents/Software/SCAagent/KEGG_mouse_2019.gmt"
}


def quick_inspect_adata(adata, max_unique=8):
    """
    Quick & compact summary of AnnData.obs for visualization purposes.
    """
    print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    print("Available layers:", list(adata.layers.keys()))
    print("Obsm keys:", list(adata.obsm.keys()))
    print()

    print("=== obs metadata (for visualization) ===")
    for col in adata.obs.columns:
        s = adata.obs[col]
        nunique = s.nunique()

        if s.dtype.name in ["category", "object"]:
            # categorical columns
            vals = s.unique()[:max_unique]
            print(f"[CAT] {col} ({nunique} categories) → {list(vals)}{' ...' if nunique > max_unique else ''}")
        elif pd.api.types.is_numeric_dtype(s):
            # numeric columns
            print(f"[NUM] {col} (range={s.min():.2f} ~ {s.max():.2f})")
    print("=======================================")

def assign_cell_categories(adata, cluster_to_category=None):
    # ensure leiden column is string (not categorical)
    adata.obs['leiden'] = adata.obs['leiden'].astype(str)

    # map cluster to category
    mapped = adata.obs['leiden'].map(cluster_to_category)

    # safe fillna
    adata.obs['cell_category'] = mapped.fillna("Unknown").astype(str)

    return adata

def assign_cell_subtype(adata, cluster_to_subtype=None):
    # ensure leiden column is string (not categorical)
    adata.obs['leiden'] = adata.obs['leiden'].astype(str)

    # map cluster to category
    mapped = adata.obs['leiden'].map(cluster_to_subtype)

    # safe fillna
    adata.obs['subtype'] = mapped.fillna("Unknown").astype(str)



def load_and_annotate_adata(path="./"):
    """
    Loads 10X Genomics data, assigns sample_label and group to adata.obs.
    Strips whitespace from group names to avoid downstream issues.
    """
    # Step 1: Load the data
    adata = sc.read_10x_mtx(
        path=path,
        var_names="gene_symbols",
        cache=True
    )

    # Step 2: Define suffix-to-label mapping
    suffix_to_label = {
        "-1": "nCoV 1", "-2": "nCoV 2", "-3": "Flu 1", "-4": "Flu 2",
        "-5": "Normal 1", "-6": "Flu 3", "-7": "Flu 4", "-8": "Flu 5",
        "-9": "nCoV 3", "-10": "nCoV 4", "-11": "nCoV 5", "-12": "nCoV 6",
        "-13": "Normal 2", "-14": "Normal 3", "-15": "nCoV 7", "-16": "nCoV 8",
        "-17": "nCoV 9", "-18": "nCoV 10", "-19": "Normal 4", "-20": "nCoV 11"
    }

    # Step 3: Helper to extract suffix from barcode
    def extract_suffix(barcode):
        return "-" + barcode.split("-")[-1]

    # Step 4: Assign sample_label
    adata.obs["sample_label"] = adata.obs_names.map(lambda x: suffix_to_label.get(extract_suffix(x), "Unknown"))

    # Step 5: Assign group (removing whitespace)
    adata.obs["group"] = adata.obs["sample_label"].str.extract(r"(\D+)").iloc[:, 0].str.strip()

    return adata


def load_and_annotate_h5_files(folder_path="/Users/silviachen/Documents/Software/SCAagent/h5_file"):
    """
    Loads all 10X `.h5` files from the given folder, processes them into a single AnnData object,
    and annotates them with their sample group ('Ctl', 'Her', 'Idio') based on BL number.
    
    Args:
        folder_path (str): Path to the folder containing `.h5` files.
    
    Returns:
        AnnData: Merged and annotated AnnData object.
    """
   

    warnings.filterwarnings("ignore", 
                            category=UserWarning,
                            module=".*")

    warnings.filterwarnings("ignore", 
                            category=UserWarning,
                            module="tqdm.auto")
    # Define sample group mapping based on BL number
    sample_group_mapping = {
        "BL6": "Ctl", "BL7": "Ctl", "BL8": "Ctl",  # Control group
        "BL1": "Her", "BL2": "Her", "BL3": "Her", "BL4": "Her", "BL5": "Her",  # Hereditary CP
        "BL9": "Idio", "BL10": "Idio", "BL11": "Idio", "BL12": "Idio"  # Idiopathic CP
    }

    # Initialize lists to store metadata and expression data
    all_metadata = []
    expression_data_list = []
    
    # Iterate through all `.h5` files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)
            
            # Extract BL number from filename
            match = re.search(r"BL\d+", filename)  # Finds "BL1", "BL2", etc.
            if match:
                bl_number = match.group()
                sample_group = sample_group_mapping.get(bl_number, "Unknown")
            else:
                print(f"⚠ Warning: Could not determine sample group for {filename}. Skipping.")
                continue  # Skip files that don't match the expected pattern
            
            # Load the dataset
            adata = sc.read_10x_h5(file_path)

            # Annotate sample group and BL number
            metadata = pd.DataFrame({
                'cell_id': adata.obs.index,  # Use cell barcodes as index
                'group': sample_group,
                'label': bl_number
            })
            metadata.set_index("cell_id", inplace=True)

            # Append metadata and expression data to lists
            all_metadata.append(metadata)
            expression_data_list.append(pd.DataFrame.sparse.from_spmatrix(adata.X, index=adata.obs.index, columns=adata.var_names))

    # Merge metadata and expression data
    if not expression_data_list:
        raise ValueError("No valid h5 files found!")

    metadata_combined = pd.concat(all_metadata, axis=0)
    expression_data_combined = pd.concat(expression_data_list, axis=0)

    # Create AnnData object
    merged_adata = sc.AnnData(X=expression_data_combined.values, obs=metadata_combined)

    # Add gene names as var
    merged_adata.var = pd.DataFrame(index=expression_data_combined.columns)
    merged_adata.var.index.name = "gene_name"

    print(f"✅ Successfully loaded {len(expression_data_list)} h5 files and merged them into a single AnnData object.")
    
    return merged_adata

    

# quality control

def filter_cells(adata, min_cells=3, min_genes=200, max_genes=50000):
    sc.pp.filter_genes(adata, min_cells=min_cells)  # Remove genes detected in <3 cells
    sc.pp.filter_cells(adata, min_genes=min_genes)  # Remove cells with <200 genes
    sc.pp.filter_cells(adata, max_genes=max_genes)  # Remove cells with >50,000 genes

def normalize_log_transform(adata, target_sum = 1e4):
    sc.pp.normalize_total(adata, target_sum = target_sum)
    sc.pp.log1p(adata)

def filter_lowqc_cells(adata, pct_counts_mt_upbound = 10, n_genes_by_counts = None, pct_counts_mt_lowbound = None):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    if n_genes_by_counts and pct_counts_mt_lowbound:
        filtered_cells = (adata.obs["n_genes_by_counts"] > n_genes_by_counts) & (adata.obs["pct_counts_mt"] > pct_counts_mt_lowbound) & (adata.obs["pct_counts_mt"] <= pct_counts_mt_upbound)
    else:
        filtered_cells = adata.obs["pct_counts_mt"] <= pct_counts_mt_upbound
    adata = adata[filtered_cells].copy()
    # temporarily placed here for DEG
    # if BL_case:
    #     adata.obs["condition"] = adata.obs["BL_number"].apply(lambda x: "Control" if x in ["BL6", "BL7", "BL8"] else "CP")
    return adata

def save_high_var_gene(adata, n_top_genes = 4000):
    sc.pp.highly_variable_genes(adata, n_top_genes = n_top_genes)
    bdata = adata[:, adata.var.highly_variable].copy()
    return bdata

import numpy as np
from sklearn.metrics import pairwise_distances

def pca_and_plot_umap(
    adata, 
    n_pcs=40, 
    n_neighbors=10, 
    resolution=0.5
):
    print(f"Set of variables used:\n - n_pcs: {n_pcs}\n - n_neighbors: {n_neighbors}\n - resolution: {resolution}\n")

    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color="leiden", size=10)


def pca_and_umap_single_run(
    adata, 
    n_pcs=40, 
    n_neighbors=10, 
    resolution=0.5
):
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    sc.tl.umap(adata)
    return adata

def ari_grid_search_clustering(
    adata,
    n_pcs_list=[10, 20, 30],
    k_list=[5, 10, 15],
    resolution=0.5,
    plot=True
):
    """
    Runs clustering across a grid of parameters and selects the most stable clustering via ARI.
    
    Returns:
        - final_adata: AnnData with best clustering and UMAP
        - ari_df: ARI matrix
        - best_label: the parameter label with highest average ARI
    """
    results = {}
    param_labels = []

    print("🔍 Running grid search...")
    for n_pcs, k in product(n_pcs_list, k_list):
        label = f"pcs={n_pcs}_k={k}"
        param_labels.append(label)
        adata_copy = adata.copy()
        adata_processed = pca_and_umap_single_run(adata_copy, n_pcs=n_pcs, n_neighbors=k, resolution=resolution)
        results[label] = adata_processed.obs["leiden"].copy()

    print("📊 Calculating ARI matrix...")
    n = len(param_labels)
    ari_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ari_matrix[i, j] = adjusted_rand_score(results[param_labels[i]], results[param_labels[j]])

    ari_df = pd.DataFrame(ari_matrix, index=param_labels, columns=param_labels)

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(ari_df, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Adjusted Rand Index (ARI) between clustering results")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    mean_ari = ari_df.mean(axis=1)
    best_label = mean_ari.idxmax()
    print(f"🏆 Best (most stable) clustering is: {best_label}")

    best_n_pcs, best_k = [int(s.split('=')[1]) for s in best_label.split('_')]
    final_adata = adata.copy()
    final_adata = pca_and_umap_single_run(final_adata, n_pcs=best_n_pcs, n_neighbors=best_k, resolution=resolution)

    # Plot final result if needed
    if plot:
        sc.pl.umap(final_adata, color="leiden", size=10)

    return final_adata, ari_df, best_label

def get_top_marker_genes(adata, groupby='leiden', method='wilcoxon', top_n=3):
    """
    Performs rank_genes_groups and returns top N marker genes per cluster.

    Parameters:
    - adata: AnnData object
    - groupby: column in adata.obs to group cells by (e.g., 'leiden')
    - method: method for DEG (default: 'wilcoxon')
    - top_n: number of top genes to extract per cluster

    Returns:
    - Dictionary with cluster name as key and top N genes as values
    """
    print(f"Ranking genes grouped by '{groupby}' using method '{method}'...")
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)

    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names  

    top_genes = {
        group: result['names'][group][:top_n].tolist()
        for group in groups
    }

    print(f"\nTop {top_n} marker genes per '{groupby}' cluster:")
    for cluster, genes in top_genes.items():
        print(f"Cluster {cluster}: {', '.join(genes)}")

    return top_genes

def annotate_by_markers(
    adata, 
    marker_gene_dict, 
    show_sample_groupinfo=True,
    show_centroid=False,
    show_raw_num=True,
    show_nearest_clusters=True,
    top_k=3
):
    import numpy as np
    from sklearn.metrics import pairwise_distances

    filtered_marker_genes = {
        cell_type: [gene for gene in genes if gene in adata.var_names]
        for cell_type, genes in marker_gene_dict.items()
    }

    # Step 1: Compute mean expression per cluster
    cluster_means = (
        adata.to_df()
        .join(adata.obs['leiden'])
        .groupby('leiden', observed=False)
        .mean()
    )

    # Step 2: Assign cell type per cluster based on marker genes
    cluster_annotations = {}
    for cluster in cluster_means.index:
        cell_type_scores = {}
        for cell_type, genes in filtered_marker_genes.items():
            if genes:
                cell_type_scores[cell_type] = cluster_means.loc[cluster, genes].mean()
        assigned_cell_type = max(cell_type_scores, key=cell_type_scores.get)
        cluster_annotations[cluster] = assigned_cell_type

    # Step 3: Map cell types back to cells
    adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_annotations)

    # Step 4: Most common sample group(s) per cluster
    top_sample_group_by_cluster = (
        adata.obs.groupby("leiden")["sample_group"]
        .agg(lambda x: x.value_counts().head(2).to_dict())
    )

    # Step 5: Centroid positions & cluster sizes
    umap_coords = adata.obsm["X_umap"]
    cluster_labels = adata.obs["leiden"]
    centroids = {}
    cluster_sizes = {}

    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]
        cluster_data = umap_coords[indices]
        centroid = cluster_data.mean(axis=0)
        centroids[cluster] = np.round(centroid, 3)
        cluster_sizes[cluster] = len(indices)

    print("Cell type annotation completed!")

    if show_sample_groupinfo:
        print("\nCluster-wise Annotation Summary:")
        cluster_ids = sorted(adata.obs['leiden'].unique(), key=int)

        # Prepare centroid array and index mapping
        if show_nearest_clusters:
            centroid_array = np.array([centroids[c] for c in cluster_ids])
            distances = pairwise_distances(centroid_array)

        for i, cluster in enumerate(cluster_ids):
            cell_type = cluster_annotations.get(cluster, "N/A")
            sample_info = top_sample_group_by_cluster.get(cluster, {})
            centroid = centroids[cluster] if show_centroid else None
            raw_num = cluster_sizes[cluster] if show_raw_num else None

            line = f"Cluster {cluster}: {cell_type}, Sample Group(s): {sample_info}"
            if show_centroid:
                line += f", Centroid: {centroid}"
            if show_raw_num:
                line += f", Size: {raw_num} cells"

            if show_nearest_clusters:
                closest_indices = np.argsort(distances[i])[1:top_k+1]
                closest_clusters = [cluster_ids[j] for j in closest_indices]
                line += f", Nearest {top_k}: {closest_clusters}"

            print(line)


def display_umap(adata, target="cell_type"):
    sc.pl.umap(adata, color=target, legend_loc="on data", title=f"UMAP colored by {target}")

    if target == "sample_group" or target == "cell_type":
        umap_coords = adata.obsm["X_umap"]
        groups = adata.obs[target]

        print(f"\nCentroid positions for '{target}':")
        for group in np.unique(groups):
            indices = np.where(groups == group)[0]
            centroid = umap_coords[indices].mean(axis=0)
            print(f"  {group} : {np.round(centroid, 3)}")

 # Step 6: Color-code BL labels for Ctl (blue), Her (red), Idio (black)
sample_colors = {"BL1": "blue", "BL2": "blue", "BL3": "blue",  # Ctl
                    "BL4": "blue", "BL5": "blue", "BL6": "red", "BL7": "red", "BL8": "red",  # Her
                    "BL9": "black", "BL10": "black", "BL11": "black", "BL12": "black"}  # Idio
 
def plot_cluster_distribution_by_BL(adata, target = "cell_type", colname = "BL_number", sample_colors = sample_colors): # or use leiden label for target
    """
    Generates a stacked bar plot showing the fraction of Leiden clusters per BL_number.
    
    Args:
        adata (AnnData): Processed AnnData object with 'leiden' clusters and 'BL_number'.
    """

    # Step 1: Count the number of cells per (BL_number, leiden cluster)
    cluster_counts = adata.obs.groupby([colname, target], observed = False).size().unstack(fill_value=0)

    # Step 2: Normalize to get fractions per BL_number
    cluster_fractions = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

    # Step 3: Sort BL numbers correctly (numerically ordered)
    cluster_fractions = cluster_fractions.sort_index()

    # Step 4: Plot using Seaborn (Stacked Bar Chart)
    plt.figure(figsize=(12, 6))
    cluster_fractions.plot(kind="bar", stacked=True, colormap="tab20", width=0.9, edgecolor="black")

    # Step 5: Customize the plot
    plt.xlabel("BL Number", fontsize=12)
    plt.ylabel("Fraction of Clusters within Each Sample", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Leiden Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

   

    for i, label in enumerate(cluster_fractions.index):
        plt.gca().get_xticklabels()[i].set_color(sample_colors.get(label, "black"))  # Default black if not found

    plt.title("Fraction of Clusters within Each BL Sample", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_cluster_frequencies(
    adata,
    cluster_key="leiden",
    group_key="sample_group",
    sample_key="BL_number"
):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1) Count cells per sample/cluster/group
    df = (
        adata.obs
        .groupby([sample_key, group_key, cluster_key], observed = False)
        .size()
        .reset_index(name="count")
    )
    
    # 2) Get total cells per sample and merge
    totals = (
        adata.obs
        .groupby(sample_key, observed = False)
        .size()
        .reset_index(name="total_count")
    )
    df = df.merge(totals, on=sample_key)
    
    # 3) Convert counts to fractions within each sample
    df["fraction"] = df["count"] / df["total_count"]
    
    # 4) Plot as a bar plot with error bars for each group
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df,
        x=cluster_key,
        y="fraction",
        hue=group_key,
        estimator=np.mean,
        errorbar='sd',         # show st. dev. error bars
        capsize=0.2
    )
    
    # 5) Overlay per-sample points (stripplot or swarmplot)
    sns.stripplot(
        data=df,
        x=cluster_key,
        y="fraction",
        hue=group_key,
        dodge=True,
        alpha=0.7,
        linewidth=1,
        edgecolor="black"
    )
    
    # 6) Avoid duplicating the legend
    handles, labels = ax.get_legend_handles_labels()
    # Only keep the first N=number_of_groups legend entries
    plt.legend(
        handles[: df[group_key].nunique()],
        labels[: df[group_key].nunique()],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0
    )
    
    # 7) Clean up labels, etc.
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average frequency of total cell clusters in each group")
    plt.xlabel("Clusters")
    plt.tight_layout()
    plt.show()


# deg and gsea
def get_enrichment_analysis(adata, target = "CP", reference = "Control", groupby = "condition", file_path = "/Users/silviachen/Documents/Software/SCAagent/h5_file/h.all.v2024.1.Hs.symbols.gmt", top_n = 10):
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata, groupby=groupby, reference=reference, method="wilcoxon")

    # Extract DEGs and sort them by log fold change
    degs_df = sc.get.rank_genes_groups_df(adata, group=target)
    degs_for_gsea = degs_df.set_index("names")["logfoldchanges"].dropna().sort_values(ascending=False)

    # Path to Hallmark Gene Set (downloaded from MSigDB)
    hallmark_gmt = file_path

    # Perform GSEA using Hallmark Gene Sets
    gsea_res = gp.prerank(
        rnk=degs_for_gsea,
        gene_sets=hallmark_gmt,  # Use the downloaded GMT file
        outdir=None,  # No output files, results stored in gsea_res
        permutation_num=1000,  # Number of permutations for significance testing
    )

    # Display enriched pathways
    print("Top Enriched Pathways in CP vs Control:")
    print(gsea_res.res2d[["Term", "ES", "NES"]].head(top_n))

def get_deg_full(
    adata,
    groupby: str = "group",
    reference: str = "Control",
    method: str = "wilcoxon",
    top_n: int = 20,
    use_raw: bool = False,
    layer: str = None,
    pts: bool = True,
    key_added: str = "rank_genes_groups",
    n_genes : int = 300
) -> dict[str, pd.DataFrame]:
    import scanpy as sc
    import pandas as pd

    # 运行差异分析
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        reference=reference,
        use_raw=use_raw,
        layer=layer,
        pts=pts,
        key_added=key_added,
        n_genes=n_genes,
    )

    result_dict = adata.uns[key_added]
    groups = result_dict["names"].dtype.names
    deg_summary = {}

    for group in groups:

        df = pd.DataFrame({
            "gene": result_dict["names"][group],
            "score": result_dict["scores"][group],
            "logFC": result_dict["logfoldchanges"][group] if "logfoldchanges" in result_dict else [None]*len(result_dict["names"][group]),
            "pval": result_dict["pvals"][group] if "pvals" in result_dict else [None]*len(result_dict["names"][group]),
            "pval_adj": result_dict["pvals_adj"][group] if "pvals_adj" in result_dict else [None]*len(result_dict["names"][group]),
        })


        # 添加表达比例（可选）
        if "pts" in result_dict:
            df["pct_group"] = result_dict["pts"].loc[df["gene"], group].values
        if "pts_rest" in result_dict:
            df["pct_rest"] = result_dict["pts_rest"].loc[df["gene"], group].values

        deg_summary[group] = df.head(top_n)
    return deg_summary

def tf_enrichment_from_adata(
    adata: sc.AnnData,
    group: Union[str, int],
    gene_set: str,
    organism: str,
    top_n: int = 200,
    outdir: str = "tf_enrichr_results",
) -> pd.DataFrame:
    # Extract top genes from adata
    rank_data = adata.uns["rank_genes_groups"]
    genes = pd.DataFrame(rank_data["names"])[group][:top_n].values.tolist()

    # Run enrichr with DoRothEA
    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_set,
        organism=organism,
        outdir=outdir,
        cutoff=0.5  # minimum combined score
    )
    result = enr.results.sort_values(
        by=["Adjusted P-value", "Combined Score"],
        ascending=[True, False]
    )
    return result

def merge_deg_summaries(
    deg_summaries: List[Dict[str, pd.DataFrame]],
    cell_types: List[str]
) -> Dict[str, List[Dict[str, object]]]:
    """
    输入：
    - deg_summaries: 每个 cell type 的 DEG 结果（多个 dict，每个 dict 只包含一个 group）
    - cell_types: 与 deg_summaries 对应的 cell type 名称列表
    
    输出：
    - gene_dict: 每个基因对应的 cell_type:stats 列表
    """
    gene_dict = {}

    for cell_type, deg in zip(cell_types, deg_summaries):
        for group, df in deg.items():
            for _, row in df.iterrows():
                gene = row["gene"]
                info = {
                    "cell_type": cell_type,
                    "pval": row.get("pval", None),
                    "score": row.get("score", None),
                    "logFC": row.get("logFC", None),
                    "pval_adj": row.get("pval_adj", None),
                }
                if gene not in gene_dict:
                    gene_dict[gene] = []
                gene_dict[gene].append(info)

    return gene_dict


def collect_tf_enrichment_details(
    adata,
    control_type: str,
    group,
    n_genes,
    species = Species
) -> Dict[str, List[Dict[str, float]]]:
    """
    For selected clusters, runs TF enrichment and collects top 3 terms with gene associations.
    If the same Term appears multiple times for a gene, retain the lower Adjusted P-value.

    Returns:
        Dict in format {gene: [ {"Term": ..., "adj_pval": ...}, ... ]}
    """
    #reset names column
    del adata.uns["rank_genes_groups"]["names"]
    get_deg_full(
                adata,
                groupby="group",          # Grouping by 'group' (Ctl, Her, Idio)
                reference=control_type,          # Using Control (Ctl) as the reference
                method="wilcoxon",        # Wilcoxon method for DEG analysis
                top_n=20,                 # Top DEGs
                use_raw=False,            # Do not use raw data
                n_genes = n_genes
            )
    from collections import defaultdict

    # Use nested dict to store best adj_pval per gene-term pair
    gene_term_map = defaultdict(dict)

    # Check if this cluster corresponds to a target cell type

    # Run enrichment
    gene_set = enrich_kmt_file_map[species]
    tf_result = tf_enrichment_from_adata(adata, group=group, gene_set=gene_set, organism = species)
    print(tf_result)
    top_rows = tf_result.iloc[:10]

    for _, row in top_rows.iterrows():
        term = row["Term"]
        adj_pval = row["Adjusted P-value"]
        genes = [g.strip() for g in row["Genes"].split(";")]

        for gene in genes:
            # If the term already exists for this gene, keep the smaller adj_pval
            if term in gene_term_map[gene]:
                gene_term_map[gene][term] = min(gene_term_map[gene][term], adj_pval)
            else:
                gene_term_map[gene][term] = adj_pval

    # Convert to required format
    final_result = {}
    for gene, term_dict in gene_term_map.items():
        final_result[gene] = [{"Term": term, "adj_pval": pval} for term, pval in term_dict.items()]

    return final_result

def topk_freq_group(adata, gene, k=5, col="subtype"):

    if gene not in adata.var_names:
        raise ValueError(f"{gene} not found in adata.var_names")

    # Gene expression vector
    expr = adata[:, gene].X
    try:
        expr = expr.toarray().flatten()
    except:
        expr = expr.flatten()

    # Construct DataFrame for grouping
    df = adata.obs.copy()
    df["expr"] = expr > 0  # binary expression

    # Calculate freq per subtype
    freq = df.groupby(col)["expr"].mean().sort_values(ascending=False)

    # Take top-k
    topk = freq.head(k)
    result = {
        f"top_{col}": k,
        "values": {sub: round(float(v), 4) for sub, v in topk.items()},
    }

    return result

def merge_deg_tf_overlap(adata, deg_dict: dict, tf_dict: dict) -> dict:
    merged = {}
    for gene in deg_dict:
        if gene in tf_dict:
            merged[gene] = {
                "subtype": topk_freq_group(adata, gene, col = "subtype"),
                "DEG": deg_dict[gene],
                "TF enrichment": tf_dict[gene]
            }
    return merged

def get_potential_gene_set(complete_list, adata, cell_types_to_analyze, group, control_type, n_genes):
    merged_dict = merge_deg_summaries(complete_list, cell_types_to_analyze)
    result = collect_tf_enrichment_details(adata, control_type, group, n_genes = n_genes)
    merged_results = merge_deg_tf_overlap(adata, deg_dict=merged_dict, tf_dict=result)
    return merged_results

def get_gene_by_disease(adata, curr_adata, curr_group, cell_types_to_analyze, control_type, n_genes = 300):
    complete_list = []
    for cell_type in cell_types_to_analyze:
        adata_temp = curr_adata[curr_adata.obs['cell_category'] == cell_type].copy()
        if adata_temp.obs["group"].value_counts().get(control_type, 0) < 5:
            continue
        deg_results = get_deg_full(
            adata_temp,
            groupby="group",          # Grouping by 'group' (Ctl, Her, Idio)
            reference=control_type,          # Using Control (Ctl) as the reference
            method="wilcoxon",        # Wilcoxon method for DEG analysis
            top_n=20,                 # Top DEGs
            use_raw=False,            # Do not use raw data
            key_added=f"deg_{cell_type}",  # Custom key for identifying result in adata
            n_genes = n_genes
        )
        complete_list.append(deg_results)
        with open(deg_tmp_file_name, "a", encoding="utf-8") as f:
            f.write(f"## {cell_type}\n")
            for grp, df in deg_results.items():
                preferred_cols = [
                    "gene", "score", "logFC", "pval", "pval_adj", "pct_group", "pct_rest"
                ]
                cols = [c for c in preferred_cols if c in df.columns]
                out_df = df[cols] if cols else df

                f.write(f"### group={grp}\n")
                out_df.to_csv(f, sep="\t", index=False)
                f.write("\n")  
            f.write("\n")      
    potential_gene_set = get_potential_gene_set(
        complete_list,           # Pass the complete list containing DEG results
        adata,                   # Original AnnData object for context
        cell_types_to_analyze,    # Cell types analyzed
        group = curr_group,
        control_type=control_type,
        n_genes = n_genes
    )
    # for gene in potential_gene_set.keys():
    #     potential_gene_set[gene]["ensembl_id"] = adata.var.loc[gene, "gene_ids"]
    folder_path = os.path.join(os.getcwd(), "potential_gene_set")

    # create folder if not exists
    os.makedirs(folder_path, exist_ok=True)

    # file name: {curr_group}.json
    output_path = os.path.join(folder_path, f"{curr_group}.json")

    # write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(potential_gene_set, f, ensure_ascii=False, indent=4)
    return potential_gene_set


def build_prerank_from_deg(adata, target: str, key_added: str = "rank_genes_groups") -> pd.Series:

    rg = adata.uns.get(key_added, None)
    if rg is None:
        raise KeyError(f"'{key_added}' not found in adata.uns. Please run rank_genes_groups / get_deg_full first.")

    if target not in rg["names"].dtype.names:
        raise ValueError(f"Target '{target}' not found in groups. Please check group name.")

    genes = pd.Index(rg["names"][target])
    if "logfoldchanges" in rg and rg["logfoldchanges"] is not None:
        vals = pd.Series(rg["logfoldchanges"][target], index=genes).astype(float)
    else:
        vals = pd.Series(rg["scores"][target], index=genes).astype(float)

    vals = vals.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return vals.sort_values(ascending=False)

def annotate_with_celltypist(adata, model_path = 'Immune_All_Low.pkl'):
    pred = celltypist.annotate(adata, model=model_path, majority_voting=True)
    adata.obs['celltypist_major'] = pred.predicted_labels.iloc[:, 0].astype(str)


def deg_for_time_series(adata, time_groups, sample_group_col="sample_group", k=10):
    """
    time_groups: list of lists, already sorted by time.
        Example: [["t1-disease1", "t2-disease1"],
                  ["t1-disease2", "t2-disease2", "t3-disease2"]]

    For each disease (each sublist), compare adjacent timepoints,
    and return top-k up/down genes.
    """

    all_results = {}

    for series in time_groups:
        # Each series is one disease/time trajectory
        series_results = {}

        # Pairwise comparison: (t1→t2), (t2→t3), ...
        for i in range(len(series) - 1):
            g1 = series[i]
            g2 = series[i + 1]

            # subset the data to only these two groups
            adata_sub = adata[adata.obs[sample_group_col].isin([g1, g2])].copy()
            adata_sub.obs["tmp_group"] = adata_sub.obs[sample_group_col].astype(str)

            # Run DEG: reference = earlier timepoint
            sc.tl.rank_genes_groups(adata_sub, groupby="tmp_group", reference=g1, method="wilcoxon")

            deg = sc.get.rank_genes_groups_df(adata_sub, group=g2)

            # top-k by logfold (positive = up-regulated)
            up = deg.sort_values("logfoldchanges", ascending=False).head(k)
            down = deg.sort_values("logfoldchanges").head(k)

            pair_key = f"{g1} → {g2}"
            series_results[pair_key] = {
                "up_genes": up[["names", "logfoldchanges", "pvals_adj"]].to_dict(orient="records"),
                "down_genes": down[["names", "logfoldchanges", "pvals_adj"]].to_dict(orient="records"),
            }

        # Add to main dictionary
        all_results[series[0].split("-")[-1]] = series_results  # group by disease name

    return all_results


from collections import defaultdict

STRING_API = "https://string-db.org/api"


def fetch_string_neighbors_clean(gene_symbol, species=9606, limit=50):
    """
    Returns a clean dict:
        { neighbor_gene_symbol : interaction_score }
    Excludes the query gene and removes duplicates.
    """

    # 1. String call
    endpoint = f"{STRING_API}/tsv/network"
    params = {
        "identifiers": gene_symbol,
        "species": species,
        "limit": limit,
        "caller_identity": "your_app"
    }

    r = requests.get(endpoint, params=params, verify=False)
    lines = r.text.strip().split("\n")

    if len(lines) < 2:
        return {}

    # 2. Parse edges
    edges = []
    for row in lines[1:]:
        cols = row.split("\t")
        edges.append({
            "protein1": cols[0],
            "protein2": cols[1],
            "score": float(cols[5]),
            "protein1_symbol": cols[2],
            "protein2_symbol": cols[3]
        })

    # 3. Identify query protein symbol
    # STRING returns the symbol in one of the two columns
    query_symbol = gene_symbol.upper()

    neighbors = {}

    for e in edges:
        # Find which side is the neighbor
        if e["protein1_symbol"].upper() == query_symbol:
            neighbor = e["protein2_symbol"]
        elif e["protein2_symbol"].upper() == query_symbol:
            neighbor = e["protein1_symbol"]
        else:
            continue  # skip weird edges not involving query gene

        # Skip the query gene itself (just in case)
        if neighbor.upper() == query_symbol:
            continue

        # Save highest score for each neighbor (dedup)
        score = e["score"]
        if neighbor not in neighbors or score > neighbors[neighbor]:
            neighbors[neighbor] = score

    return neighbors




def fetch_gene_summary(gene_ensemeble_id):
    """
    gene_input: Ensembl ID (e.g., "ENSG00000112367") or gene symbol (e.g., "CCR6")
    returns: dict containing summary, uniprot info, GO terms, etc.
    """

    url = "https://mygene.info/v3/query"

    # Accept both symbol and Ensembl ID
    params = {
        "q": gene_ensemeble_id,
        "fields": "summary,uniprot,go.BP",
        "species": "human"
    }

    r = requests.get(url, params=params, verify=False)
    data = r.json()

    if "hits" not in data or len(data["hits"]) == 0:
        return {"error": "No gene found"}

    hit = data["hits"][0]

    # ----- Extract summary from NCBI -----
    summary = hit.get("summary", None)

    # ----- Extract UniProt function text -----
    uniprot = None
    if "uniprot" in hit:
        # UniProt may return dict with 'func' or 'comments'
        if isinstance(hit["uniprot"], dict):
            uniprot = hit["uniprot"].get("func") or hit["uniprot"].get("comments")

    # ----- Extract GO Biological Process -----
    go_terms = set()
    if "go" in hit and "BP" in hit["go"]:
        for term in hit["go"]["BP"]:
            if isinstance(term, dict) and "term" in term:
                go_terms.add(term["term"])
            elif isinstance(term, str):
                go_terms.add(term)

    return {
        "query": gene_ensemeble_id,
        "summary": summary,
        "uniprot_function": uniprot,
        "go_biological_process": list(go_terms)
    }



API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

def fetch_target_data(ensembl_id):
    query = """
    query targetQuery($ensembl_id: String!) {
      target(ensemblId: $ensembl_id) {
        approvedSymbol

        knownDrugs {
          rows {
            phase
          }
        }

        associatedDiseases {
          rows {
            disease { id name }
            datasourceScores { id score }
          }
        }
      }
    }
    """

    variables = {"ensembl_id": ensembl_id}
    r = requests.post(API_URL, json={"query": query, "variables": variables}, verify=False)
    resp = r.json()

    if "errors" in resp:
        raise ValueError(resp["errors"])

    return resp["data"]["target"]


def is_drug_target(ensemble_id):
    t = fetch_target_data(ensemble_id)
    reasons = []

    # -----------------------------
    # 1 — knownDrugs block may be None
    # -----------------------------
    known_block = t.get("knownDrugs", {}) or {}
    known_rows = known_block.get("rows", []) or []

    if len(known_rows) > 0:
        reasons.append(f"Known drug entries found ({len(known_rows)} entries)")
        return True, reasons

    # -----------------------------
    # 2 — associatedDiseases may be None
    # -----------------------------
    disease_block = t.get("associatedDiseases", {}) or {}
    disease_rows = disease_block.get("rows", []) or []

    for row in disease_rows:
        datasource_list = row.get("datasourceScores", []) or []
        for ds in datasource_list:
            if ds.get("id") == "chembl" and ds.get("score", 0) > 0:
                reasons.append(
                    f"ChEMBL drug evidence for disease {row['disease']['name']}"
                )
                return True, reasons

    # -----------------------------
    # 3 — No evidence found
    # -----------------------------
    reasons.append("No known drugs and no ChEMBL drug evidence")
    return False, reasons


import os
import json

def get_filtered_gene_list(adata, gene_list):
    result = []

    output_file = "drug_target_filtered.json"

    # Load existing json if exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            record_dict = json.load(f)
    else:
        record_dict = {}

    # Process each gene
    for gene_symbol in gene_list:
        ensemble_id = adata.var.loc[gene_symbol, "gene_ids"]
        is_drug, reasoning = is_drug_target(ensemble_id)

        # (1) Add to return list only if NOT drug target
        if not is_drug:
            result.append(gene_symbol)

        # (2) Write to JSON regardless of drug status
        record_dict[gene_symbol] = {
            "ensemble_id": ensemble_id,
            "is_drug_target": is_drug,
            "reasoning": reasoning if reasoning else ""
        }

    # Write JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(record_dict, f, indent=4, ensure_ascii=False)

    return result
