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

    # Compute UMAP centroids
    # umap_coords = adata.obsm["X_umap"]
    # cluster_labels = adata.obs["leiden"]

    # centroids = {}
    # cluster_sizes = {}

    # for cluster in np.unique(cluster_labels):
    #     indices = np.where(cluster_labels == cluster)[0]
    #     cluster_data = umap_coords[indices]
    #     centroid = cluster_data.mean(axis=0)
    #     centroids[cluster] = centroid
    #     cluster_sizes[cluster] = len(indices)

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
                     "BL4": "blue", "BL5": "blue", "BL6": "red", "BL7": "red", "BL8": "red",  # Her
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