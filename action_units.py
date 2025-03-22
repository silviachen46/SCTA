class ActionUnit:
    def __init__(self, name, instruction):
        self.name = name
        self.instruction = instruction

    def __str__(self):
        return f"Name: {self.name}\nInstruction: {self.instruction}"
    
class ActionUnitWithContext(ActionUnit):
    def __init__(self, name, instruction, action_unit):
        super().__init__(name, instruction) 
        self.action_unit = action_unit
        self.codes = {}
        self.errors = {}
        self.curr_result = None
        self.debug_num = 0
        self.run_num = 0

    def add_code(self, code):
        self.codes.append(code)

    def add_error(self, error):
        self.errors.append(error)

general_action_units = {
    "general_unit": """
      You are to conduct Single Cell analysis on the given data. 
      Before start, use import warnings
      warnings.filterwarnings("ignore", category=UserWarning) to filter the warning

      You should first quality control & filter the data,
      PCA and compute UMAP,
      annotate with given functions and marker set marker_genes from markers.py,
      plot your UMAP by cell_type, sample_group, and leiden cluster,
      plot cluster distribution by BL and plot by frequency with given function.
      do DEG and GSEA.

      and then extract the t cell from previous adata into adata_t. t cell is with "T_cells" in its adata.obs['cell_type'].
      redo log transform, PCA and compute UMAP,
      
      annotate with given functions and marker set marker_genes_t from markers.py,

      plot your UMAP by cell_type, sample_group, and leiden cluster,
      plot cluster distribution by BL and then plot by frequency with given function.

      All required path has been specified for you as default parameter, you do not need to modify them. You are allowed to tune other given params.
      You are given the following helper functions, and you should fully leverage them to conduct analysis.
      
      

      You can import them using "from utils_agent import load_and_annotate_h5_files, filter_cells, normalize_log_transform, filter_lowqc_cells, save_high_var_gene, pca_and_plot_umap, annoatate_by_markers"
      You will need marker dict for annotation. Use from markers import marker_genes, marker_genes_t for the function.
      Important: All return types are marked for you. If a return type is an adata, you should write as adata = func(adata). Otherwise you will miss important information.

      def load_and_annotate_h5_files(folder_path="/Users/silviachen/Documents/Software/SCAagent/h5_file") -> adata;

      def filter_cells(adata, min_cells=3, min_genes=200, max_genes=50000) -> None:
      def normalize_log_transform(adata, target_sum = 1e4) -> None:
      def filter_lowqc_cells(adata, pct_counts_mt_upbound = 10, n_genes_by_counts = None, pct_counts_mt_lowbound = None) -> adata:
      def save_high_var_gene(adata, n_top_genes = 4000) -> adata:
      
      def pca_and_plot_umap(adata, n_pcs = 40, n_neighbors = 10, resolution = 0.5) -> None:
      def annoatate_by_markers(adata, marker_gene_dict) -> None

      def display_umap(adata, target = "cell_type") -> None: # cell_type, BLnumber for target
      def plot_cluster_distribution_by_BL(adata, target = "cell_type") -> None: # or use 'leiden' label for target
      def plot_cluster_frequencies(
    adata,
    cluster_key="leiden",
    group_key="sample_group",
    sample_key="BL_number"
) -> None:

   def get_deg(adata, top_n = 20, groupby = "condition", reference = "Control") -> None:
   def get_enrichment_analysis(adata, target = "CP", reference = "Control", groupby = "condition", file_path = "/Users/silviachen/Documents/Software/SCAagent/h5_file/h.all.v2024.1.Hs.symbols.gmt", top_n = 10) -> None:
      
      
"""
}
# general_action_units = {
    
#    "load_and_annotate": """
#         You are given a folder `"agent/data_file"` containing multiple `.tsv` files. These files follow the naming pattern:

# Example filenames:
# - "GSM3576396_C9_R_cell-gene_UMI_table.tsv"
# - "GSM3576413_U4_pBMC_cell-gene_UMI_table.tsv"


# Your task is to:
# 1. **Iterate through all `.tsv` files** in `"agent/data_file"`, filtering only those that match the pattern: *_cell-gene_UMI_table.tsv
# 2. **Extract metadata from filenames**:
# - `sample_id`: The second element in the filename after splitting by `_`
#   (e.g., `"C9"` from `"GSM3576396_C9_R_cell-gene_UMI_table.tsv"`).
# - `marker_type`: The **first character** of `sample_id` (e.g., `"C"` from `"C9"`, `"U"` from `"U4"`).
# 3. **Load each `.tsv` file into a Pandas DataFrame**, ensuring:
# - It is **read with tab (`\t`) as the delimiter**.
# - The first column (cell IDs) is used as the **index**.
# 4. **Store the metadata separately** in a Pandas DataFrame:
# - Columns: `cell_id`, `sample_id`, `marker_type`
# - Set `"cell_id"` as the index.
# 5. **Concatenate all metadata and expression data** across multiple files into a **single dataset**.
# 6. **Create an `AnnData` object** where:
# - `X` contains the expression data matrix.
# - `obs` contains the metadata.
# - `var` contains the gene names as a DataFrame with `"gene_name"` as the index.
# 7. **Ensure unique gene names using**:
# ```python
# adata.var_names_make_unique()
# Verify the final AnnData object by printing it.
# Your final output should be a single AnnData object containing all the loaded expression data and metadata."""

#     ,
    
#         "Preprocess and Filter scRNA-seq Data": """
# Assume `adata` is already loaded. Use Scanpy's built-in functions for efficient preprocessing while ensuring correct handling of sparse data and avoiding unintended view modifications.

# 1. **Compute QC Metrics:**  
#    - Identify mitochondrial genes by checking if gene names start with `"MT."` and store this information in `adata.var["mt"]`.  
#    - use sc function to compute total UMI counts, detected gene counts and mitochondrial RNA percentage, Ensure these values are stored. 

# 2. **Filter Low-Quality Cells:**  
#    - Retain only cells with `n_genes_by_counts > 400` and mitochondrial content strictly greater than 0.5 percent and at most 10%.  

# 4. **Normalize & Log Transform:**  
#    - Normalize total UMI counts per cell to **1,000,000**.  
#    - Apply a logarithmic transformation.

# 5. **Select and Subset Highly Variable Genes:**  
#    - Subset the dataset to retain only the selected highly variable genes.

# Ensure correct indexing, prevent NaN-related issues, and maintain computational efficiency. Do not print any final confirmation messages.  

# """


#     ,
    
#         "clustering and visualization": """You can assume we already have the AnnData in the variable adata. Run PCA on the AnnData object with proper number of components chosen by you using sc.tl.pca(). Keep the number of PCs you choose in n_pcs.
#       Plot the explained variance ratio to confirm the selection of PCs using sc.pl.pca_variance_ratio(). Construct the neighborhood graph with n_pcs chosen by you and proper number of neighbors using sc.pp.neighbors(). 
#       Perform clustering with Leiden algorithm at resolution 2.0 using sc.tl.leiden(). 
#       Run UMAP for dimensionality reduction and visualize the clustering results by coloring the UMAP plot based on Leiden clusters, with point size set to 10.""",

#       "annotation":"""Assume `adata` is already clustered and contains Leiden cluster assignments. Use Scanpy's built-in functions to identify marker genes and annotate cell types based on known markers.

# 1. **Identify Marker Genes for Each Cluster:**  
#    - Perform differential expression analysis between clusters specifying `"leiden"` and `"wilcoxon"` as the statistical method.  
#    - Visualize the top differentially expressed genes per cluster.  

# 2. **Count Cells per Cell Type:**  
#    - Retrieve and print the distribution of cells across clusters by counting occurrences of each `"cell_type"` in `adata.obs`.  

# 3. **Define Marker Gene Sets for Cell Type Annotation:**  
#    - Create a dictionary of marker genes corresponding to known immune cell types, including 'T cells', 'B cells', 'Myeloid cells','NK cells'. You should also follow this format when storing annotations.
#    - Ensure that gene names match those in `adata.var_names` to avoid missing markers.  

# 4. **Assign Cell Types to Clusters:**  
#    - Iterate over unique Leiden clusters in `adata.obs["leiden"]`.  
#    - Extract the subset of cells belonging to each cluster and compute the average expression of marker genes for each predefined cell type.  
#    - Assign each cluster the cell type with the highest average expression among its marker genes.  

# 5. **Store Cluster Annotations:**  
#    - Create a mapping between Leiden cluster IDs and their most likely cell type labels.  
#    - Apply this mapping to `adata.obs["leiden"]` to generate a new column `"cell_type"`, storing the inferred identity of each cluster.  

# Ensure consistency in gene name matching, handle missing markers gracefully, and verify results by inspecting the `"cell_type"` column in `adata.obs`.  
# """,

# "B cell": """Read adata from this file "annotated_single_cell_data.h5ad". Now you have adata and cell type in adata.obs["cell_type"] with one of the four types: 'T cells', 'B cells', 'Myeloid cells','NK cells'. Now extract B cell data in to a new adata_b. Do PCA, construct new neighbour graphs, and perform Leiden Clustering for B cells.
# Assign cluster label, compute UMAP for B cells, and visualize them. After clustering is complete, group the observation metadata in `adata_b.obs` by both cluster labels and marker types to generate a summary of cell distributions.  
# Use a pivot table to reshape the grouped data, ensuring that marker types are represented as separate columns. Plot a stacked bar chart where clusters are on the x-axis, and the number of cells in each marker type is represented by stacked bars. Assign red color to 'U' marker type and blue color to 'C' marker type for visualization. Customize axis labels, title, and legend for clarity, ensuring that x-axis labels are rotated for readability. Adjust layout to optimize figure appearance.  
# """,

# "B cell":
# """Read adata from this file "annotated_single_cell_data.h5ad". You now have adata where adata.obs["cell_type"] contains four types: 'T cells', 'B cells', 'Myeloid cells', and 'NK cells'. Extract B cell data into adata_b and perform PCA. Construct a neighbor graph and perform Leiden clustering to identify B cell clusters. Store cluster labels in adata_b.obs for easy reference.

# Compute UMAP fsor visualization and generate a UMAP plot where clusters are colored according to their Leiden labels.s

# Once clustering is complete, summarize the distribution of cells across clusters and marker types using adata_b.obs. Group by cluster labels and marker types, then reshape the grouped data using a pivot table so that marker types appear as separate columns.

# Use the pivot table to generate a stacked bar chart, placing clusters on the x-axis and stacking the number of cells by marker type. Assign colors to marker types ('U' in red, 'C' in blue`). Use built-in plotting utilities to visualize the chart efficiently. Ensure axis labels, the title, and the legend are clearly labeled, with x-axis labels rotated for readability. Optimize figure layout for clarity."""
# ,

    

# }




# general_action_units = {
    
#    "load_and_annotate": """
#         You are given a folder `"agent/data_file"` containing multiple `.tsv` files. These files follow the naming pattern:

# Example filenames:
# - "GSM3576396_C9_R_cell-gene_UMI_table.tsv"
# - "GSM3576413_U4_pBMC_cell-gene_UMI_table.tsv"


# Your task is to:
# 1. **Iterate through all `.tsv` files** in `"agent/data_file"`, filtering only those that match the pattern: *_cell-gene_UMI_table.tsv
# 2. **Extract metadata from filenames**:
# - `sample_id`: The second element in the filename after splitting by `_`
#   (e.g., `"C9"` from `"GSM3576396_C9_R_cell-gene_UMI_table.tsv"`).
# - `marker_type`: The **first character** of `sample_id` (e.g., `"C"` from `"C9"`, `"U"` from `"U4"`).
# 3. **Load each `.tsv` file into a Pandas DataFrame**, ensuring:
# - It is **read with tab (`\t`) as the delimiter**.
# - The first column (cell IDs) is used as the **index**.
# 4. **Store the metadata separately** in a Pandas DataFrame:
# - Columns: `cell_id`, `sample_id`, `marker_type`
# - Set `"cell_id"` as the index.
# 5. **Concatenate all metadata and expression data** across multiple files into a **single dataset**.
# 6. **Create an `AnnData` object** where:
# - `X` contains the expression data matrix.
# - `obs` contains the metadata.
# - `var` contains the gene names as a DataFrame with `"gene_name"` as the index.
# 7. **Ensure unique gene names using**:
# ```python
# adata.var_names_make_unique()
# Verify the final AnnData object by printing it.
# Your final output should be a single AnnData object containing all the loaded expression data and metadata."""

#     ,
    
#         "Preprocess and Filter scRNA-seq Data": """
# Assume `adata` is already loaded. Use Scanpy's built-in functions for efficient preprocessing while ensuring correct handling of sparse data and avoiding unintended view modifications.

# 1. **Compute QC Metrics:**  
#    - Identify mitochondrial genes by checking if gene names start with `"MT."` and store this information in `adata.var["mt"]`.  
#    - use sc function to compute total UMI counts, detected gene counts and mitochondrial RNA percentage, Ensure these values are stored. 

# 2. **Filter Low-Quality Cells:**  
#    - Retain only cells with `n_genes_by_counts > 400` and mitochondrial content strictly greater than 0.5 percent and at most 10%.  

# 4. **Normalize & Log Transform:**  
#    - Normalize total UMI counts per cell to **1,000,000**.  
#    - Apply a logarithmic transformation.

# 5. **Select and Subset Highly Variable Genes:**  
#    - Subset the dataset to retain only the selected highly variable genes.

# Ensure correct indexing, prevent NaN-related issues, and maintain computational efficiency. Do not print any final confirmation messages.  

# """


#     ,
    
#         "clustering and visualization": """You can assume we already have the AnnData in the variable adata. Run PCA on the AnnData object with proper number of components chosen by you using sc.tl.pca(). Keep the number of PCs you choose in n_pcs.
#       Plot the explained variance ratio to confirm the selection of PCs using sc.pl.pca_variance_ratio(). Construct the neighborhood graph with n_pcs chosen by you and proper number of neighbors using sc.pp.neighbors(). 
#       Perform clustering with Leiden algorithm at resolution 2.0 using sc.tl.leiden(). 
#       Run UMAP for dimensionality reduction and visualize the clustering results by coloring the UMAP plot based on Leiden clusters, with point size set to 10.""",

#       "annotation":"""Assume `adata` is already clustered and contains Leiden cluster assignments. Use Scanpy's built-in functions to identify marker genes and annotate cell types based on known markers.

# 1. **Identify Marker Genes for Each Cluster:**  
#    - Perform differential expression analysis between clusters using `sc.tl.rank_genes_groups()`, specifying `"leiden"` as the grouping variable and `"wilcoxon"` as the statistical method.  
#    - Visualize the top differentially expressed genes per cluster with `sc.pl.rank_genes_groups()`, displaying at least 10 genes per cluster.  

# 2. **Count Cells per Cell Type:**  
#    - Retrieve the distribution of cells across clusters by counting occurrences of each `"cell_type"` in `adata.obs`.  
#    - Print these counts to verify the representation of different cell populations.  

# 3. **Define Marker Gene Sets for Cell Type Annotation:**  
#    - Create a dictionary of marker genes corresponding to known immune cell types, including 'T cells', 'B cells', 'Myeloid cells','NK cells'. You should also follow this format when storing annotations.
#    - Ensure that gene names match those in `adata.var_names` to avoid missing markers.  

# 4. **Assign Cell Types to Clusters:**  
#    - Iterate over unique Leiden clusters in `adata.obs["leiden"]`.  
#    - Extract the subset of cells belonging to each cluster and compute the average expression of marker genes for each predefined cell type.  
#    - Assign each cluster the cell type with the highest average expression among its marker genes.  

# 5. **Store Cluster Annotations:**  
#    - Create a mapping between Leiden cluster IDs and their most likely cell type labels.  
#    - Apply this mapping to `adata.obs["leiden"]` to generate a new column `"cell_type"`, storing the inferred identity of each cluster.  

# Ensure consistency in gene name matching, handle missing markers gracefully, and verify results by inspecting the `"cell_type"` column in `adata.obs`.  
# """,

# "B cell": """Read adata from this file "annotated_single_cell_data.h5ad". Now you have adata and cell type in adata.obs["cell_type"] with one of the four types: 'T cells', 'B cells', 'Myeloid cells','NK cells'. Now extract B cell data in to a new adata_b. Do PCA, construct new neighbour graphs, and perform Leiden Clustering for B cells.
# Assign cluster label, compute UMAP for B cells, and visualize them. After clustering is complete, group the observation metadata in `adata_b.obs` by both cluster labels and marker types to generate a summary of cell distributions.  
# Use a pivot table to reshape the grouped data, ensuring that marker types are represented as separate columns. Plot a stacked bar chart where clusters are on the x-axis, and the number of cells in each marker type is represented by stacked bars. Assign red color to 'U' marker type and blue color to 'C' marker type for visualization. Customize axis labels, title, and legend for clarity, ensuring that x-axis labels are rotated for readability. Adjust layout to optimize figure appearance.  
# """,
# "B cell":
# """Read adata from this file "annotated_single_cell_data.h5ad". You now have adata where adata.obs["cell_type"] contains four types: 'T cells', 'B cells', 'Myeloid cells', and 'NK cells'. Extract B cell data into adata_b and perform PCA. Construct a neighbor graph and perform Leiden clustering to identify B cell clusters. Store cluster labels in adata_b.obs for easy reference.

# Compute UMAP fsor visualization and generate a UMAP plot where clusters are colored according to their Leiden labels.s

# Once clustering is complete, summarize the distribution of cells across clusters and marker types using adata_b.obs. Group by cluster labels and marker types, then reshape the grouped data using a pivot table so that marker types appear as separate columns.

# Use the pivot table to generate a stacked bar chart, placing clusters on the x-axis and stacking the number of cells by marker type. Assign colors to marker types ('U' in red, 'C' in blue`). Use built-in plotting utilities to visualize the chart efficiently. Ensure axis labels, the title, and the legend are clearly labeled, with x-axis labels rotated for readability. Optimize figure layout for clarity."""
# ,

    

# }