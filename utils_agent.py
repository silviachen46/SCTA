import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
import os, re
import warnings
from anndata import ImplicitModificationWarning
# load and mark

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
                'sample_group': sample_group,
                'BL_number': bl_number
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
    adata.obs["condition"] = adata.obs["BL_number"].apply(lambda x: "Control" if x in ["BL6", "BL7", "BL8"] else "CP")
    return adata

def save_high_var_gene(adata, n_top_genes = 4000):
    sc.pp.highly_variable_genes(adata, n_top_genes = n_top_genes)
    bdata = adata[:, adata.var.highly_variable].copy()
    return bdata

# pca and umap

def pca_and_plot_umap(adata, n_pcs = 40, n_neighbors = 10, resolution = 0.5):
    print(f"Set of variables used:\n - n_pcs: {n_pcs}\n - n_neighbors: {n_neighbors}\n - resolution: {resolution}\n")
   
    # sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")

    sc.pl.pca_variance_ratio(adata, log=True)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)  # Adjusted k for stability

    sc.tl.leiden(adata, resolution=resolution, flavor="igraph",  n_iterations=2)  # Balanced resolution

    sc.tl.umap(adata)
    
    sc.pl.umap(adata, color="leiden", size=10)

# annotate

def annoatate_by_markers(adata, marker_gene_dict):
    filtered_marker_genes = {}
    for cell_type, genes in marker_gene_dict.items():
        matching_genes = [gene for gene in genes if gene in adata.var_names]
        # if not matching_genes:
        #     print(f"⚠ Warning: No matching genes found for {cell_type}. Check gene names.")
        # else:
        #     print(f"{cell_type}: {matching_genes}")
        filtered_marker_genes[cell_type] = matching_genes
    cluster_means = (
        adata.to_df()
        .join(adata.obs['leiden'])  # Add cluster labels
        .groupby('leiden', observed = False)  # Group by cluster
        .mean()  # Compute mean expression per cluster
    )
    cluster_annotations = {}
    for cluster in cluster_means.index:
        cell_type_scores = {}
        for cell_type, genes in filtered_marker_genes.items():
            if genes:  # Only consider non-empty gene lists
                cell_type_scores[cell_type] = cluster_means.loc[cluster, genes].mean()
        # Assign the cell type with the highest mean expression
        assigned_cell_type = max(cell_type_scores, key=cell_type_scores.get)
        cluster_annotations[cluster] = assigned_cell_type
    # Step 4: Map cluster annotations to individual cells
    adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_annotations)

    print("Cell type annotation completed!")
    print(adata.obs[['leiden', 'cell_type']].drop_duplicates().sort_values('leiden'))

# plotting

def display_umap(adata, target = "cell_type"): # cell_type, BLnumber, label of cltr v.s. CP for target
    sc.pl.umap(adata, color=target, legend_loc="on data", title="Cell Type Annotations")


def plot_cluster_distribution_by_BL(adata, target = "cell_type"): # or use leiden label for target
    """
    Generates a stacked bar plot showing the fraction of Leiden clusters per BL_number.
    
    Args:
        adata (AnnData): Processed AnnData object with 'leiden' clusters and 'BL_number'.
    """

    # Step 1: Count the number of cells per (BL_number, leiden cluster)
    cluster_counts = adata.obs.groupby(["BL_number", target], observed = False).size().unstack(fill_value=0)

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

    # Step 6: Color-code BL labels for Ctl (blue), Her (red), Idio (black)
    sample_colors = {"BL1": "blue", "BL2": "blue", "BL3": "blue",  # Ctl
                     "BL4": "red", "BL5": "red", "BL6": "red", "BL7": "red", "BL8": "red",  # Her
                     "BL9": "black", "BL10": "black", "BL11": "black", "BL12": "black"}  # Idio

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

def get_deg(adata, top_n = 20, groupby = "condition", reference = "Control"):
    
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", reference=reference)

    deg_results = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(top_n)  # Get top 20 genes

    # Print top 20 DEGs
    print("Top " + str(top_n) + " DEGs in CP vs Control:")
    print(deg_results)

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